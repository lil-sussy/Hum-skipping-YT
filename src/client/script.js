
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
	/*
  humskip.user.js

  Overview:
  - Captures audio from the main <video> element on YouTube.
  - Records short chunks every CHUNK_INTERVAL_SEC seconds and sends them
    to a Python server endpoint (e.g. `POST /infer_chunk`) using GM_xmlhttpRequest
    to avoid page CORS and mixed-content blocking.
  - Receives JSON with labeled timestamps relative to chunk start and video time.
  - Maps relative intervals to absolute video timestamps using explicit math,
    merges adjacent ranges, and performs smooth skips.

  Notes about network:
  - Using window.fetch from an https page (YouTube) to an http://localhost:PORT
    is blocked by mixed-content rules.
  - Tampermonkey's GM_xmlhttpRequest runs with extension privileges and can
    make requests to http://127.0.0.1 and LAN or web servers. Use GM_xmlhttpRequest
    for maximum compatibility.
  - If your server runs HTTPS, you may use fetch as well.

  Configure below constants to your environment.
*/
	// -------------------- Configuration --------------------
	const SERVER_URL = "http://127.0.0.1:8887/infer_chunk"; // change to your server endpoint
	const CHUNK_INTERVAL_SEC = 0.1; // how often to send chunks (seconds)
	const CHUNK_DURATION_SEC = 1.0; // chunk length in seconds (could equal interval)
	const MERGE_TOLERANCE_SEC = 0.15; // merge tolerance when joining nearby ranges
	const ENABLE_AUTO_SKIP_DEFAULT = true;
	const LOG_MAX = 200; // max kept logs in localStorage
	const UI_OPACITY = 0.85;
	// -------------------------------------------------------

	// Minimal UI injection for status and controls
	GM_addStyle(`
  #humskip-ui {
    position: fixed;
    right: 12px;
    bottom: 80px;
    z-index: 2147483647;
    background: rgba(0,0,0,0.6);
    color: white;
    padding: 8px 10px;
    border-radius: 8px;
    font-family: Arial, sans-serif;
    font-size: 12px;
    opacity: ${UI_OPACITY};
  }
  #humskip-ui button { margin-left:8px; font-size:12px; }
  #humskip-segments { max-height: 120px; overflow:auto; margin-top:6px; font-size:11px; }
`);

	// UI state
	let ui = null;
	function createUI() {
		ui = document.createElement("div");
		ui.id = "humskip-ui";
		ui.innerHTML = `
    <div>
      <strong>HumSkip</strong>
      <label style="margin-left:8px;">
        <input type="checkbox" id="humskip-autoskip" /> Auto-skip
      </label>
      <button id="humskip-toggle">Start</button>
    </div>
    <div id="humskip-status">idle</div>
    <div id="humskip-segments"></div>
  `;
		document.body.appendChild(ui);

		const chk = ui.querySelector("#humskip-autoskip");
		chk.checked = ENABLE_AUTO_SKIP_DEFAULT;
		ui.querySelector("#humskip-toggle").addEventListener("click", async () => {
			if (captureActive) {
				stopCapture();
				ui.querySelector("#humskip-toggle").textContent = "Start";
			} else {
				const started = await startCaptureFlow();
				if (started) ui.querySelector("#humskip-toggle").textContent = "Stop";
			}
		});
	}

	// -------------------- Logging --------------------
	function pushLog(o) {
    console.warn("[Tamper monkey - Hum aware skip YT] log :");
    console.log(o)
		const logs = JSON.parse(localStorage.getItem("humskip_logs") || "[]");
		logs.push({ t: Date.now(), ...o });
		while (logs.length > LOG_MAX) logs.shift();
		localStorage.setItem("humskip_logs", JSON.stringify(logs));
	}
	// -------------------- Utility math (explicit) --------------------
	/*
  Mapping math:

  Let sr be the sample rate (samples/sec) used by server/model.
  Let frame_sec be CHUNK_DURATION_SEC.
  Let video_time = t0 (seconds) at which chunk started recording.

  If server returns a label with relative times r_start..r_end (seconds inside chunk),
  absolute timestamps are computed as:

    abs_start = t0 + r_start
    abs_end   = t0 + r_end

  This is derived from:
    start_sample_index = round(t0 * sr)
    rel_start_sample = round(r_start * sr)
    => absolute_sample = start_sample_index + rel_start_sample
    => abs_start = absolute_sample / sr = t0 + r_start

  For scheduling: compare video.currentTime (now) with abs_start..abs_end.
*/

	// -------------------- Core mechanics --------------------
	let captureActive = false;
	let mediaRecorder = null;
	let mediaStream = null;
	let pendingRanges = []; // {start, end, label, score}
	let lastSentChunkId = 0;

	// Merge overlapping / nearby ranges (same label)
	function mergeRanges(ranges) {
		if (!Array.isArray(ranges) || ranges.length === 0) return [];
		const sorted = ranges.slice().sort((a, b) => a.start - b.start);
		const out = [];
		let cur = { ...sorted[0] };
		for (let i = 1; i < sorted.length; i++) {
			const r = sorted[i];
			if (r.label === cur.label && r.start <= cur.end + MERGE_TOLERANCE_SEC) {
				// extend
				cur.end = Math.max(cur.end, r.end);
				cur.score = Math.max(cur.score || 0, r.score || 0);
			} else {
				out.push(cur);
				cur = { ...r };
			}
		}
		out.push(cur);
		return out;
	}

	// When new ranges are received, merge them into pendingRanges and attempt skip
	function pushRanges(newRanges) {
		pendingRanges = mergeRanges(pendingRanges.concat(newRanges));
		updateUISegments();
		attemptSkip(); // immediate attempt if currently inside a range
	}

	// Update UI list
	function updateUISegments() {
		if (!ui) return;
		const segDiv = ui.querySelector("#humskip-segments");
		segDiv.innerHTML = "";
		for (const r of pendingRanges.slice(0, 20)) {
			const el = document.createElement("div");
			el.textContent = `${r.label} ${r.start.toFixed(2)}→${r.end.toFixed(2)} s (score:${(r.score || 0).toFixed(2)})`;
			segDiv.appendChild(el);
		}
	}

	// Locate YouTube video element (main)
	function findYouTubeVideo() {
		// video with largest area on page (heuristic)
		const vids = Array.from(document.querySelectorAll("video"));
		if (vids.length === 0) return null;
		let best = vids[0];
		for (const v of vids) {
			const a = v.videoWidth * v.videoHeight;
			if (a > best.videoWidth * best.videoHeight) best = v;
		}
		return best;
	}

	// Smooth skip behavior: pause -> seek -> play
	async function doSmoothSkip(video, targetTime) {
		try {
			video.pause();
			// small fade-out could be implemented here by reducing volume
			video.currentTime = targetTime;
			// micro-wait to ensure seek takes effect
			await new Promise((r) => setTimeout(r, 40));
			video.play();
			pushLog({ event: "skipped_to", target: targetTime });
		} catch (e) {
			console.warn("HumSkip: smooth skip failed", e);
			pushLog({ event: "skip_error", error: String(e) });
		}
	}

	// Try to skip if current time is inside a pending range
	function attemptSkip() {
		const video = findYouTubeVideo();
		if (!video) return;
		const now = video.currentTime;
		for (const r of pendingRanges) {
			if (now >= r.start && now < r.end) {
				const target = r.end + 0.02; // safety offset
				// Only skip if auto-skip enabled in UI
				const auto = ui ? ui.querySelector("#humskip-autoskip").checked : ENABLE_AUTO_SKIP_DEFAULT;
				if (auto) doSmoothSkip(video, target);
				break;
			}
		}
	}

	// -------------------- Networking: GM_xmlhttpRequest wrapper --------------------
	function postChunkToServer({ blob, videoTime, chunkId }) {
		return new Promise((resolve, reject) => {
			// Build form data manually with multipart boundary since GM_xmlhttpRequest needs raw body
			const boundary = "----HumSkipBoundary" + Math.random().toString(36).slice(2);
			const CRLF = "\r\n";
			const metaParts = [`--${boundary}`, `Content-Disposition: form-data; name="video_time"`, ``, String(videoTime)];
			const blobHeader = [`--${boundary}`, `Content-Disposition: form-data; name="audio_blob"; filename="chunk_${chunkId}.webm"`, `Content-Type: ${blob.type || "application/octet-stream"}`, ``].join(CRLF);

			const footer = CRLF + `--${boundary}--` + CRLF;

			// Convert headers and meta to ArrayBuffer
			const encoder = new TextEncoder();
			const headerBuf = encoder.encode(metaParts.join(CRLF) + CRLF + blobHeader + CRLF);

			// Assemble final body as ArrayBuffer: headerBuf + blob + footerBuf
			const reader = new FileReader();
			reader.onload = function () {
				const blobBuf = reader.result;
				const footerBuf = encoder.encode(footer);
				// Concatenate
				const totalLen = headerBuf.byteLength + blobBuf.byteLength + footerBuf.byteLength;
				const tmp = new Uint8Array(totalLen);
				tmp.set(new Uint8Array(headerBuf), 0);
				tmp.set(new Uint8Array(blobBuf), headerBuf.byteLength);
				tmp.set(new Uint8Array(footerBuf), headerBuf.byteLength + blobBuf.byteLength);

				// Send via GM_xmlhttpRequest with proper headers
				GM_xmlhttpRequest({
					method: "POST",
					url: SERVER_URL,
					binary: true,
					data: tmp.buffer,
					headers: {
						"Content-Type": `multipart/form-data; boundary=${boundary}`,
					},
					onload: function (res) {
						try {
							if (res.status >= 200 && res.status < 300) {
								const json = JSON.parse(res.responseText);
								resolve(json);
							} else {
								reject(new Error("Server returned status " + res.status));
							}
						} catch (e) {
							reject(e);
						}
					},
					onerror: function (err) {
						reject(err);
					},
					ontimeout: function () {
						reject(new Error("timeout"));
					},
				});
			};
			reader.onerror = function (err) {
				reject(err);
			};
			// Read blob as ArrayBuffer
			reader.readAsArrayBuffer(blob);
		});
	}

	// -------------------- Capture flow --------------------
	async function startCaptureFlow() {
		if (captureActive) {
      pushLog("Capture is already active");
			return false;
		}

		const video = findYouTubeVideo();
		if (!video) {
			alert("HumSkip: no video element found on page.");
			return false;
		}

		createUIIfMissing();
		ui.querySelector("#humskip-status").textContent = "Initializing capture...";


		// Request user gesture for audio capture. We'll attempt captureStream first.
		try {
			// Attempt captureStream first
			let mediaStream = null;
			if (video.captureStream) {
				try {
					mediaStream = video.captureStream();
				} catch (e) {
					console.warn("Initial captureStream attempt failed:", e);
					// Try again after user gesture
					await new Promise((resolve) => setTimeout(resolve, 500));
					try {
						mediaStream = video.captureStream();
					} catch (e) {
						console.warn("Second captureStream attempt failed:", e);
						mediaStream = null;
					}
				}
			}

			if (!mediaStream) {
				ui.querySelector("#humskip-status").textContent = "captureStream unavailable. Attempting WebAudio fallback.";
				console.log("Falling back to WebAudio API");
				return startWebAudioCapture(video);
			}

			// Initialize MediaRecorder
			// const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : MediaRecorder.isTypeSupported("audio/ogg;codecs=opus") ? "audio/ogg;codecs=opus" : "audio/webm";
			// Initialize MediaRecorder with supported MIME type
			const supportedTypes = [];
			const testTypes = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/webm", "audio/ogg"];

			for (const type of testTypes) {
				if (MediaRecorder.isTypeSupported(type)) {
					supportedTypes.push(type);
				}
			}

			if (supportedTypes.length === 0) {
				throw new Error("No supported MIME types found");
			}

			// Try each supported type until one works
			for (const mimeType of supportedTypes) {
				try {
					mediaRecorder = new MediaRecorder(mediaStream, { mimeType });
					console.log(`Successfully initialized MediaRecorder with MIME type: ${mimeType}`);
					break;
				} catch (error) {
					console.warn(`Failed to initialize with ${mimeType}:`, error);
					continue;
				}
			}

			if (!mediaRecorder) {
				throw new Error("Failed to initialize MediaRecorder with any supported MIME type");
			}

			// Setup event handlers
			mediaRecorder.ondataavailable = async (event) => {
				if (!event.data || event.data.size === 0) return;

				const chunkId = ++lastSentChunkId;
				const videoTime = findYouTubeVideo()?.currentTime || 0;

				try {
					ui.querySelector("#humskip-status").textContent = `sending chunk ${chunkId} @ ${videoTime.toFixed(2)}s`;

					const json = await postChunkToServer({
						blob: event.data,
						videoTime,
						chunkId,
					});

					const labels = (json.labels || []).map((label) => ({
						label: label.label || "unknown",
						start: (label.start || 0) + videoTime,
						end: (label.end || CHUNK_DURATION_SEC) + videoTime,
						score: label.score || 0,
					}));

					pushRanges(labels);
					ui.querySelector("#humskip-status").textContent = `received ${labels.length} segments`;

					pushLog({
						event: "recv_segments",
						chunkId,
						count: labels.length,
					});
				} catch (error) {
					ui.querySelector("#humskip-status").textContent = `send error: ${error.message || error}`;
					pushLog({
						event: "send_error",
						chunkId,
						error: String(error),
					});
				}
			};

			mediaRecorder.onerror = (error) => {
				ui.querySelector("#humskip-status").textContent = "recorder error";
				pushLog({
					event: "recorder_error",
					error: String(error),
				});
				cleanupCapture();
				// Try WebAudio fallback
				if (!captureActive) {
					startWebAudioCapture(video);
				}
			};

			mediaRecorder.start(Math.round(CHUNK_INTERVAL_SEC * 1000));
			captureActive = true;
			ui.querySelector("#humskip-status").textContent = "capturing via MediaRecorder";

			return true;
		} catch (error) {
			ui.querySelector("#humskip-status").textContent = "capture failed";
			pushLog({
				event: "capture_failed",
				error: String(error),
			});
			cleanupCapture();
			return false;
		}
		if (!mediaStream) {
			// fallback: ask user to enable a small overlay button to click on the page (user gesture)
			// Some browsers require video.play() user gesture to allow captureStream.
			// Ask user to click play/pause once, then try again.
			ui.querySelector("#humskip-status").textContent = "Requesting user gesture for capture...";
			pushLog({ event: "need_user_gesture" });
			// Wait briefly and retry
			await new Promise((r) => setTimeout(r, 500));
			try {
				mediaStream = video.captureStream ? video.captureStream() : null;
			} catch (e) {
				mediaStream = null;
			}
			if (!mediaStream) {
				ui.querySelector("#humskip-status").textContent = "captureStream unavailable. Attempting WebAudio fallback.";
				// WebAudio fallback: use createMediaElementSource and ScriptProcessor or AudioWorklet.
				try {
					mediaStream = null; // we'll use AudioContext directly below
				} catch (e) {
					console.warn("WebAudio fallback failed", e);
				}
			}
		}

		// If we have a mediaStream, use MediaRecorder approach (encodes with Opus/webm)
		if (mediaStream && typeof MediaRecorder !== "undefined") {
			try {
				// Example MIME: audio/webm;codecs=opus - supported in modern browsers
				const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : MediaRecorder.isTypeSupported("audio/ogg;codecs=opus") ? "audio/ogg;codecs=opus" : mediaStream.getAudioTracks()[0]?.contentType || "audio/webm";
				mediaRecorder = new MediaRecorder(mediaStream, { mimeType: mime });
			} catch (e) {
				console.warn("MediaRecorder init failed:", e);
				mediaRecorder = null;
			}
		}

		if (mediaRecorder) {
			// chunk handling
			mediaRecorder.ondataavailable = async (ev) => {
				if (!ev.data || ev.data.size === 0) return;
				const chunkId = ++lastSentChunkId;
				const videoTime = findYouTubeVideo()?.currentTime || 0;
				ui.querySelector("#humskip-status").textContent = `sending chunk ${chunkId} @ ${videoTime.toFixed(2)}s`;
				pushLog({ event: "send_chunk", chunkId, videoTime, size: ev.data.size });

				// Send with GM_xmlhttpRequest to server
				try {
					const json = await postChunkToServer({ blob: ev.data, videoTime, chunkId });
					// Expected JSON example:
					// { "labels": [{"label":"hum","start":0.0,"end":1.0,"score":0.9}, ...], "frame_start": videoTime }
					const labels = (json.labels || []).map((l) => ({
						label: l.label || "unknown",
						start: (l.start || 0) + videoTime,
						end: (l.end || CHUNK_DURATION_SEC) + videoTime,
						score: l.score || 0,
					}));
					pushRanges(labels);
					ui.querySelector("#humskip-status").textContent = `received ${labels.length} segments`;
					pushLog({ event: "recv_segments", chunkId, count: labels.length });
				} catch (err) {
					ui.querySelector("#humskip-status").textContent = `send error: ${err.message || err}`;
					pushLog({ event: "send_error", chunkId, error: String(err) });
				}
			};

			mediaRecorder.onstart = () => {
				ui.querySelector("#humskip-status").textContent = "recording...";
				pushLog({ event: "recorder_start" });
			};
			mediaRecorder.onerror = (e) => {
				ui.querySelector("#humskip-status").textContent = "recorder error";
				pushLog({ event: "recorder_error", error: String(e) });
			};
			mediaRecorder.start(Math.round(CHUNK_INTERVAL_SEC * 1000)); // timeslice in ms
			captureActive = true;
			ui.querySelector("#humskip-status").textContent = "capturing via MediaRecorder";
			return true;
		}

		// If we reach here, MediaRecorder isn't available—use AudioContext + ScriptProcessor (lower-level)
		try {
			const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
			const source = audioCtx.createMediaElementSource(video);
			const bufferSize = 4096;
			const proc = audioCtx.createScriptProcessor(bufferSize, 2, 2);
			let pcmBuffer = [];
			const sr = audioCtx.sampleRate;
			const neededSamples = Math.round(CHUNK_DURATION_SEC * sr);

			proc.onaudioprocess = (e) => {
				const left = e.inputBuffer.getChannelData(0);
				const right = e.inputBuffer.numberOfChannels > 1 ? e.inputBuffer.getChannelData(1) : left;
				for (let i = 0; i < left.length; i++) {
					const mono = 0.5 * (left[i] + right[i]);
					pcmBuffer.push(mono);
				}
				if (pcmBuffer.length >= neededSamples) {
					// Build a Float32Array for exact chunk length
					const chunkArray = pcmBuffer.slice(0, neededSamples);
					pcmBuffer = pcmBuffer.slice(neededSamples);
					// Convert float32 PCM [-1..1] to 16-bit PCM for server compatibility (if your server expects this)
					const ab = new ArrayBuffer(chunkArray.length * 2);
					const dv = new DataView(ab);
					for (let i = 0; i < chunkArray.length; i++) {
						let s = Math.max(-1, Math.min(1, chunkArray[i]));
						dv.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
					}
					const blob = new Blob([ab], { type: "application/octet-stream" });
					const chunkId = ++lastSentChunkId;
					const videoTime = findYouTubeVideo()?.currentTime || 0;
					ui.querySelector("#humskip-status").textContent = `sending raw chunk ${chunkId} @ ${videoTime.toFixed(2)}s`;
					postChunkToServer({ blob, videoTime, chunkId })
						.then((json) => {
							const labels = (json.labels || []).map((l) => ({
								label: l.label || "unknown",
								start: (l.start || 0) + videoTime,
								end: (l.end || CHUNK_DURATION_SEC) + videoTime,
								score: l.score || 0,
							}));
							pushRanges(labels);
							ui.querySelector("#humskip-status").textContent = `received ${labels.length} segments`;
						})
						.catch((err) => {
							ui.querySelector("#humskip-status").textContent = `send error: ${err.message || err}`;
							pushLog({ event: "send_error", chunkId, error: String(err) });
						});
				}
			};

			source.connect(proc);
			proc.connect(audioCtx.destination); // keep audio audible
			captureActive = true;
			ui.querySelector("#humskip-status").textContent = "capturing via ScriptProcessor (raw PCM)";
			pushLog({ event: "raw_capture_start", sr: audioCtx.sampleRate });
			return true;
		} catch (e) {
			ui.querySelector("#humskip-status").textContent = "capture failed";
			pushLog({ event: "capture_failed", error: String(e) });
			return false;
		}
	}

	function stopCapture() {
		if (!captureActive) return;
		captureActive = false;
		if (mediaRecorder && mediaRecorder.state !== "inactive") {
			try {
				mediaRecorder.stop();
			} catch (e) {}
			mediaRecorder = null;
		}
		if (mediaStream) {
			try {
				for (const t of mediaStream.getTracks()) t.stop();
			} catch (e) {}
			mediaStream = null;
		}
		ui.querySelector("#humskip-status").textContent = "stopped";
		pushLog({ event: "capture_stopped" });
	}

	// Ensure UI existence
	function createUIIfMissing() {
		if (!ui) createUI();
	}

	// Auto-attempt attach on page load (but require user to press Start)
	(function bootstrap() {
		createUIIfMissing();
		// Add keyboard shortcut for undo last skip (u)
		window.addEventListener("keydown", (e) => {
			if (e.key === "u") {
				// try to step back a small amount
				const video = findYouTubeVideo();
				if (!video) return;
				const back = Math.max(0, video.currentTime - 1.0);
				video.currentTime = back;
				pushLog({ event: "undo_skip", to: back });
			}
		});

		// Poll attemptSkip periodically in case transitions occur between chunk arrivals
		setInterval(() => {
			if (!captureActive) return;
			attemptSkip();
		}, 250);

		// Small auto-start attempt if you want (commented)
		// startCaptureFlow();
	})();
})();
