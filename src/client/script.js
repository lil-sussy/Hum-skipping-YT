
(function () {
	"use strict";
	if (window.trustedTypes && window.trustedTypes.createPolicy && !window.trustedTypes.defaultPolicy) {
		window.trustedTypes.createPolicy("default", {
			createHTML: (string) => string,
			// Optional, only needed for script (url) tags
			//,createScriptURL: string => string
			//,createScript: string => string,
		});
	}

	/* ========= CONFIG ========= */
	// Edit this regex to control which URLs trigger the server call.
	// Default matches YouTube watch pages with a v= parameter.
	const URL_REGEX = /youtube\.com\/watch\?v=([^&]+)/i;

	// Local server base
	const SERVER_BASE = "http://127.0.0.1:8887";

	// Poll interval in ms
	const POLL_MS = 2000;

	// UI settings
	let AUTO_SKIP_ENABLED = true; // default, saved in session storage
	/* ========================== */

	// small helper for GM_xmlhttpRequest wrapped in Promise
	function gmFetch(options) {
		return new Promise((resolve, reject) => {
			const defaultOpts = {
				method: options.method || "GET",
				url: options.url,
				headers: options.headers || {},
				responseType: options.responseType || "json",
				data: options.body || null,
			};

			GM_xmlhttpRequest({
				method: defaultOpts.method,
				url: defaultOpts.url,
				headers: defaultOpts.headers,
				data: defaultOpts.data,
				responseType: defaultOpts.responseType,
				onload: function (res) {
					// GM will parse JSON if responseType json; otherwise res.responseText
					resolve(res);
				},
				onerror: function (err) {
					reject(err);
				},
				ontimeout: function () {
					reject(new Error("timeout"));
				},
			});
		});
	}

	// Utility: extract video id if match
	function extractVideoIdFromUrl(url) {
		const m = URL_REGEX.exec(url);
		if (m) return m[1];
		return null;
	}

	// UI: floating control panel
	function createUIPanel() {
		const tpl = `
      <div id="humskip-panel" style="font-family: Arial, Helvetica, sans-serif; position: fixed; right: 12px; bottom: 12px; z-index: 1000000;">
        <div id="humskip-card" style="background: rgba(0,0,0,0.75); color: white; padding: 8px 12px; border-radius: 8px; width: 220px; box-shadow: 0 6px 18px rgba(0,0,0,0.4);">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
            <strong style="font-size:13px;">HumSkip</strong>
            <button id="humskip-close" title="Hide panel" style="background:transparent;border:0;color:#ccc;cursor:pointer;">✕</button>
          </div>
          <div style="margin-top:8px;font-size:12px;">
            <label style="display:flex; align-items:center; gap:8px;">
              <input id="humskip-toggle" type="checkbox" /> Auto-skip hum & silence
            </label>
            <div id="humskip-status" style="margin-top:8px;font-size:12px;color:#ddd;">Status: idle</div>
            <div id="humskip-info" style="margin-top:6px;font-size:11px;color:#aaa;"></div>
            <div style="margin-top:8px; display:flex; gap:6px;">
              <button id="humskip-refresh" style="flex:1;padding:6px;border-radius:6px;border:0;cursor:pointer;">Fetch timeline</button>
              <button id="humskip-manual-skip" style="flex:1;padding:6px;border-radius:6px;border:0;cursor:pointer;">Skip hum now</button>
            </div>
          </div>
        </div>
      </div>
    `;
		const wrapper = document.createElement("div");
		wrapper.innerHTML = tpl;
		document.body.appendChild(wrapper);

		// wire events
		const toggle = document.getElementById("humskip-toggle");
		toggle.checked = sessionStorage.getItem("humskip_auto") !== "false";
		AUTO_SKIP_ENABLED = toggle.checked;
		toggle.addEventListener("change", () => {
			AUTO_SKIP_ENABLED = toggle.checked;
			sessionStorage.setItem("humskip_auto", AUTO_SKIP_ENABLED ? "true" : "false");
			updateStatus("Auto-skip " + (AUTO_SKIP_ENABLED ? "enabled" : "disabled"));
		});

		document.getElementById("humskip-close").addEventListener("click", () => {
			document.getElementById("humskip-panel").style.display = "none";
		});

		document.getElementById("humskip-refresh").addEventListener("click", () => {
			const vid = getCurrentVideoId();
			if (vid) {
				sendVideoToServerAndWatch(vid);
			} else {
				updateStatus("No video id found");
			}
		});

		document.getElementById("humskip-manual-skip").addEventListener("click", () => {
			// immediate attempt to skip if inside a hum/silence range
			attemptSkipImmediate();
		});

		return {
			setStatus: updateStatus,
			setInfo: (s) => (document.getElementById("humskip-info").innerText = s),
		};

		function updateStatus(text) {
			const el = document.getElementById("humskip-status");
			el.innerText = "Status: " + text;
		}
	}

	const ui = createUIPanel();

	// job store in script
	let currentJobId = null;
	let currentVideoId = null;
	let timeline = null; // the JSON content with segments
	let skipRanges = []; // merged hum/silence ranges: {start, end}

	// helper to get current video element and id
	function findYouTubeVideo() {
		const video = document.querySelector("video");
		return video;
	}
	function getCurrentVideoId() {
		const url = window.location.href;
		return extractVideoIdFromUrl(url);
	}

  function getYouTubeCookieHeader() {
		return document.cookie
			.split(";")
			.map((c) => c.trim())
			.join("; ");
	}

	// Collect essential headers to mimic browser request
	function getYouTubeClientHeaders() {
		return {
			"User-Agent": navigator.userAgent,
			"Accept-Language": navigator.language || "en-US,en;q=0.9",
			Referer: window.location.href,
			Cookie: getYouTubeCookieHeader(),
			// Add other headers as needed, e.g. 'Origin', 'Connection', 'Accept', ...
      "Origin": navigator.Origin,
      "Connection": navigator.connection ? navigator.connection.effectiveType : "unknown",
      "Accept": navigator.accept || "*/*",
		};
	}

	// call server to start job and poll
	async function sendVideoToServerAndWatch(videoId) {
		try {
			ui.setStatus("Submitting video to server...");
			// POST /infer_video with JSON {url: current_url}
			const payload = JSON.stringify({ url: window.location.href, video_id: videoId, client_headers: JSON.stringify(getYouTubeClientHeaders()) });

			const postRes = await gmFetch({
				method: "POST",
				url: SERVER_BASE + "/infer_video",
				headers: { "Content-Type": "application/json" },
				body: payload,
				responseType: "json",
			});

			// parse response
			const postBody = postRes.response ? postRes.response : JSON.parse(postRes.responseText || "{}");
			if (!postBody || !postBody.job_id) {
				ui.setStatus("Server did not return job_id");
				console.warn("unexpected response", postRes);
				return;
			}
			currentJobId = postBody.job_id;
			ui.setInfo(`job=${currentJobId}`);
			ui.setStatus("Processing started; polling...");
			// poll
			await pollJob(currentJobId);
		} catch (err) {
			console.error(err);
			ui.setStatus("Error posting video to server");
		}
	}

	// poll job status endpoint until done or error
	async function pollJob(jobId) {
		try {
			while (true) {
				const res = await gmFetch({
					method: "GET",
					url: `${SERVER_BASE}/jobs/${jobId}/status`,
					responseType: "json",
				});
				const body = res.response ? res.response : JSON.parse(res.responseText || "{}");
				if (!body) {
					ui.setStatus("Bad status response");
					return;
				}
				if (body.status === "done") {
					ui.setStatus("Done. fetching timeline...");
					await fetchTimeline(jobId, /*videoId*/ getCurrentVideoId());
					return;
				} else if (body.status === "failed") {
					ui.setStatus("Failed: " + (body.error || "unknown"));
					return;
				} else {
					ui.setStatus(`Processing (${body.progress || "?"})`);
				}
				await new Promise((r) => setTimeout(r, POLL_MS));
			}
		} catch (err) {
			console.error("poll job error", err);
			ui.setStatus("Polling error");
		}
	}

	// fetch timeline JSON (GET /jobs/{id}/result)
	async function fetchTimeline(jobId, videoId) {
		try {
			const res = await gmFetch({
				method: "GET",
				url: `${SERVER_BASE}/jobs/${jobId}/result`,
				responseType: "json",
			});
			const body = res.response ? res.response : JSON.parse(res.responseText || "{}");
			if (!body || !body.segments) {
				ui.setStatus("No timeline returned");
				return;
			}
			timeline = body;
			currentVideoId = videoId || timeline.video_id;
			ui.setStatus("Timeline received");
			ui.setInfo(`segments=${timeline.segments.length}`);
			// build skipRanges: only hum & silence
			skipRanges = timeline.segments.filter((s) => s.label === "hum" || s.label === "silence").map((s) => ({ start: s.start, end: s.end }));
			// ensure sorted, merge small gaps
			skipRanges.sort((a, b) => a.start - b.start);
			skipRanges = mergeRanges(skipRanges, 0.15);
			// attach skip watcher
			attachSkipWatcher();
		} catch (err) {
			console.error("fetchTimeline error", err);
			ui.setStatus("Failed to fetch timeline");
		}
	}

	function mergeRanges(ranges, tol) {
		if (!ranges || ranges.length === 0) return [];
		const out = [];
		let cur = { ...ranges[0] };
		for (let i = 1; i < ranges.length; i++) {
			const r = ranges[i];
			if (r.start <= cur.end + tol) {
				cur.end = Math.max(cur.end, r.end);
			} else {
				out.push(cur);
				cur = { ...r };
			}
		}
		out.push(cur);
		return out;
	}

	// Attach watch loop to video element to auto skip when inside a range
	let watcherAttached = false;
	function attachSkipWatcher() {
		if (watcherAttached) return;
		const video = findYouTubeVideo();
		if (!video) {
			ui.setStatus("No video element found to attach watcher");
			return;
		}

		// Setup WebAudio gain node to do smooth fades
		let audioCtx;
		let sourceNode;
		let gainNode;
		try {
			audioCtx = new (window.AudioContext || window.webkitAudioContext)();
			sourceNode = audioCtx.createMediaElementSource(video);
			gainNode = audioCtx.createGain();
			sourceNode.connect(gainNode);
			gainNode.connect(audioCtx.destination);
		} catch (e) {
			console.warn("WebAudio setup failed; skipping fade functionality", e);
			audioCtx = null;
			gainNode = null;
		}

		// main timeupdate loop (not too often)
		let lastCheck = 0;
		video.addEventListener("timeupdate", () => {
			const now = performance.now();
			if (now - lastCheck < 120) return; // check ~8 times/sec
			lastCheck = now;
			if (!AUTO_SKIP_ENABLED) return;
			attemptSkip(video, gainNode);
		});

		watcherAttached = true;
		ui.setStatus("Watcher attached");
	}

	// Attempt to skip immediately if in a skip range
	async function attemptSkipImmediate() {
		const video = findYouTubeVideo();
		if (!video) return;
		attemptSkip(video, null);
	}

	// Find if video.currentTime inside a skip range and perform a smooth skip
	// fadeDuration in seconds
	let lastSkipAt = -1;
	async function attemptSkip(videoEl, gainNode) {
		if (!skipRanges || skipRanges.length === 0) return;
		const t = videoEl.currentTime;
		// prevent repeated skips too frequently; allow once per 0.2s
		if (Math.abs(t - lastSkipAt) < 0.1) return;
		for (const r of skipRanges) {
			if (t >= r.start && t < r.end) {
				// skip to r.end + small offset
				const target = Math.min(r.end + 0.02, videoEl.duration || r.end + 0.02);
				lastSkipAt = t;
				// do small fade out if possible
				try {
					if (gainNode && gainNode.gain) {
						// fade out 160ms
						const now = (gainNode.context && gainNode.context.currentTime) || 0;
						gainNode.gain.cancelScheduledValues(now);
						gainNode.gain.setValueAtTime(gainNode.gain.value, now);
						gainNode.gain.linearRampToValueAtTime(0.0001, now + 0.16);
						// seek after 120ms to make crossfade smoother
						setTimeout(() => {
							try {
								videoEl.currentTime = target;
							} catch (e) {
								videoEl.currentTime = target;
							}
							// fade back in
							const now2 = (gainNode.context && gainNode.context.currentTime) || 0;
							gainNode.gain.cancelScheduledValues(now2);
							gainNode.gain.setValueAtTime(0.0001, now2);
							gainNode.gain.linearRampToValueAtTime(1.0, now2 + 0.2);
						}, 120);
					} else {
						// fallback: quick pause/seek/play
						videoEl.pause();
						videoEl.currentTime = target;
						// small delay to ensure seek took effect
						setTimeout(() => {
							try {
								videoEl.play();
							} catch (e) {}
						}, 50);
					}
				} catch (err) {
					console.error("skip error", err);
				}
				break;
			}
		}
	}

	// auto-trigger when URL matches
	function maybeTriggerOnLoad() {
		const currentUrl = window.location.href;
		const vid = extractVideoIdFromUrl(currentUrl);
		if (!vid) {
			// do nothing
			return;
		}
		// small delay to let page settle
		setTimeout(() => {
			// auto-submit when matched
			sendVideoToServerAndWatch(vid);
		}, 800);
	}

	// initial call
	maybeTriggerOnLoad();

	// Listen for navigation changes (YouTube SPA navigation)
	// Two approaches: popstate and pushState hijack. Observe location changes.
	let lastHref = location.href;
	new MutationObserver(() => {
		if (location.href !== lastHref) {
			lastHref = location.href;
			// reset timeline if video changed
			const newVid = extractVideoIdFromUrl(lastHref);
			if (newVid && newVid !== currentVideoId) {
				currentVideoId = newVid;
				timeline = null;
				skipRanges = [];
				ui.setInfo("video changed; fetching new timeline...");
				sendVideoToServerAndWatch(newVid);
			}
		}
	}).observe(document, { subtree: true, childList: true });

	// small helper to attempt skip without passing gainNode (for manual)
	window.HumSkip = { forceFetch: () => sendVideoToServerAndWatch(getCurrentVideoId()) };
})();
