// ==UserScript==
// @name         HumSkip (Tampermonkey) - capture YouTube audio and skip hums/silences
// @namespace    https://example.com/humskip
// @version      0.1.0
// @description  Capture short audio chunks from YouTube, send to local/remote server, receive timestamps, and smoothly skip unwanted parts.
// @author       Jettsy
// @match        *://*.youtube.com/*
// @grant        GM_xmlhttpRequest
// @grant        GM_notification
// @grant        GM_addStyle
// @connect      127.0.0.1
// @require      file:///home/Jettsy/Desktop/devlab/Hum-skipping-YT/src/client/script.js
// @connect      localhost
// @connect      127.0.0.1:8887
// @run-at       document-idle
// ==/UserScript==
