require("./constants.cjs");
let node_stream_web = require("node:stream/web");
let node_stream = require("node:stream");
//#region src/ssr/transformStreamWithRouter.ts
function transformReadableStreamWithRouter(router, routerStream) {
	return transformStreamWithRouter(router, routerStream);
}
function transformPipeableStreamWithRouter(router, routerStream) {
	return node_stream.Readable.fromWeb(transformStreamWithRouter(router, node_stream.Readable.toWeb(routerStream)));
}
var BODY_END_TAG = "</body>";
var HTML_END_TAG = "</html>";
var MIN_CLOSING_TAG_LENGTH = 4;
var DEFAULT_SERIALIZATION_TIMEOUT_MS = 6e4;
var DEFAULT_LIFETIME_TIMEOUT_MS = 6e4;
var textEncoder = new TextEncoder();
/**
* Finds the position just after the last valid HTML closing tag in the string.
*
* Valid closing tags match the pattern: </[a-zA-Z][\w:.-]*>
* Examples: </div>, </my-component>, </slot:name.nested>
*
* @returns Position after the last closing tag, or -1 if none found
*/
function findLastClosingTagEnd(str) {
	const len = str.length;
	if (len < MIN_CLOSING_TAG_LENGTH) return -1;
	let i = len - 1;
	while (i >= MIN_CLOSING_TAG_LENGTH - 1) {
		if (str.charCodeAt(i) === 62) {
			let j = i - 1;
			while (j >= 1) {
				const code = str.charCodeAt(j);
				if (code >= 97 && code <= 122 || code >= 65 && code <= 90 || code >= 48 && code <= 57 || code === 95 || code === 58 || code === 46 || code === 45) j--;
				else break;
			}
			const tagNameStart = j + 1;
			if (tagNameStart < i) {
				const startCode = str.charCodeAt(tagNameStart);
				if (startCode >= 97 && startCode <= 122 || startCode >= 65 && startCode <= 90) {
					if (j >= 1 && str.charCodeAt(j) === 47 && str.charCodeAt(j - 1) === 60) return i + 1;
				}
			}
		}
		i--;
	}
	return -1;
}
function transformStreamWithRouter(router, appStream, opts) {
	const serializationAlreadyFinished = router.serverSsr?.isSerializationFinished() ?? false;
	const initialBufferedHtml = router.serverSsr?.takeBufferedHtml();
	if (serializationAlreadyFinished && !initialBufferedHtml) {
		let cleanedUp = false;
		let controller;
		let isStreamClosed = false;
		let lifetimeTimeoutHandle;
		const cleanup = () => {
			if (cleanedUp) return;
			cleanedUp = true;
			if (lifetimeTimeoutHandle !== void 0) {
				clearTimeout(lifetimeTimeoutHandle);
				lifetimeTimeoutHandle = void 0;
			}
			router.serverSsr?.cleanup();
		};
		const safeClose = () => {
			if (isStreamClosed) return;
			isStreamClosed = true;
			try {
				controller?.close();
			} catch {}
		};
		const safeError = (error) => {
			if (isStreamClosed) return;
			isStreamClosed = true;
			try {
				controller?.error(error);
			} catch {}
		};
		const lifetimeMs = opts?.lifetimeMs ?? DEFAULT_LIFETIME_TIMEOUT_MS;
		lifetimeTimeoutHandle = setTimeout(() => {
			if (!cleanedUp && !isStreamClosed) {
				console.warn(`SSR stream transform exceeded maximum lifetime (${lifetimeMs}ms), forcing cleanup`);
				safeError(/* @__PURE__ */ new Error("Stream lifetime exceeded"));
				cleanup();
			}
		}, lifetimeMs);
		const stream = new node_stream_web.ReadableStream({
			start(c) {
				controller = c;
			},
			cancel() {
				isStreamClosed = true;
				cleanup();
			}
		});
		(async () => {
			const reader = appStream.getReader();
			try {
				while (true) {
					const { done, value } = await reader.read();
					if (done) break;
					if (cleanedUp || isStreamClosed) return;
					controller?.enqueue(value);
				}
				if (cleanedUp || isStreamClosed) return;
				router.serverSsr?.setRenderFinished();
				safeClose();
				cleanup();
			} catch (error) {
				if (cleanedUp) return;
				console.error("Error reading appStream:", error);
				router.serverSsr?.setRenderFinished();
				safeError(error);
				cleanup();
			} finally {
				reader.releaseLock();
			}
		})().catch((error) => {
			if (cleanedUp) return;
			console.error("Error in stream transform:", error);
			safeError(error);
			cleanup();
		});
		return stream;
	}
	let stopListeningToInjectedHtml;
	let stopListeningToSerializationFinished;
	let serializationTimeoutHandle;
	let lifetimeTimeoutHandle;
	let cleanedUp = false;
	let controller;
	let isStreamClosed = false;
	const textDecoder = new TextDecoder();
	let pendingRouterHtml = initialBufferedHtml ?? "";
	let leftover = "";
	let pendingClosingTags = "";
	const MAX_LEFTOVER_CHARS = 2048;
	let isAppRendering = true;
	let streamBarrierLifted = false;
	let serializationFinished = serializationAlreadyFinished;
	function safeEnqueue(chunk) {
		if (isStreamClosed) return;
		if (typeof chunk === "string") controller.enqueue(textEncoder.encode(chunk));
		else controller.enqueue(chunk);
	}
	function safeClose() {
		if (isStreamClosed) return;
		isStreamClosed = true;
		try {
			controller.close();
		} catch {}
	}
	function safeError(error) {
		if (isStreamClosed) return;
		isStreamClosed = true;
		try {
			controller.error(error);
		} catch {}
	}
	/**
	* Cleanup with guards; must be idempotent.
	*/
	function cleanup() {
		if (cleanedUp) return;
		cleanedUp = true;
		try {
			stopListeningToInjectedHtml?.();
			stopListeningToSerializationFinished?.();
		} catch {}
		stopListeningToInjectedHtml = void 0;
		stopListeningToSerializationFinished = void 0;
		if (serializationTimeoutHandle !== void 0) {
			clearTimeout(serializationTimeoutHandle);
			serializationTimeoutHandle = void 0;
		}
		if (lifetimeTimeoutHandle !== void 0) {
			clearTimeout(lifetimeTimeoutHandle);
			lifetimeTimeoutHandle = void 0;
		}
		pendingRouterHtml = "";
		leftover = "";
		pendingClosingTags = "";
		router.serverSsr?.cleanup();
	}
	const stream = new node_stream_web.ReadableStream({
		start(c) {
			controller = c;
		},
		cancel() {
			isStreamClosed = true;
			cleanup();
		}
	});
	function flushPendingRouterHtml() {
		if (!pendingRouterHtml) return;
		safeEnqueue(pendingRouterHtml);
		pendingRouterHtml = "";
	}
	function appendRouterHtml(html) {
		if (!html) return;
		pendingRouterHtml += html;
	}
	/**
	* Finish only when app done and serialization complete.
	*/
	function tryFinish() {
		if (isAppRendering || !serializationFinished) return;
		if (cleanedUp || isStreamClosed) return;
		if (serializationTimeoutHandle !== void 0) {
			clearTimeout(serializationTimeoutHandle);
			serializationTimeoutHandle = void 0;
		}
		const decoderRemainder = textDecoder.decode();
		if (leftover) safeEnqueue(leftover);
		if (decoderRemainder) safeEnqueue(decoderRemainder);
		flushPendingRouterHtml();
		if (pendingClosingTags) safeEnqueue(pendingClosingTags);
		safeClose();
		cleanup();
	}
	const lifetimeMs = opts?.lifetimeMs ?? DEFAULT_LIFETIME_TIMEOUT_MS;
	lifetimeTimeoutHandle = setTimeout(() => {
		if (!cleanedUp && !isStreamClosed) {
			console.warn(`SSR stream transform exceeded maximum lifetime (${lifetimeMs}ms), forcing cleanup`);
			safeError(/* @__PURE__ */ new Error("Stream lifetime exceeded"));
			cleanup();
		}
	}, lifetimeMs);
	if (!serializationAlreadyFinished) {
		stopListeningToInjectedHtml = router.subscribe("onInjectedHtml", () => {
			if (cleanedUp || isStreamClosed) return;
			const html = router.serverSsr?.takeBufferedHtml();
			if (!html) return;
			if (isAppRendering || leftover || pendingClosingTags) appendRouterHtml(html);
			else safeEnqueue(html);
		});
		stopListeningToSerializationFinished = router.subscribe("onSerializationFinished", () => {
			serializationFinished = true;
			tryFinish();
		});
	}
	(async () => {
		const reader = appStream.getReader();
		try {
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				if (cleanedUp || isStreamClosed) return;
				const text = value instanceof Uint8Array ? textDecoder.decode(value, { stream: true }) : String(value);
				const chunkString = leftover ? leftover + text : text;
				if (!streamBarrierLifted) {
					if (chunkString.includes("$tsr-stream-barrier")) {
						streamBarrierLifted = true;
						router.serverSsr?.liftScriptBarrier();
					}
				}
				if (pendingClosingTags) {
					pendingClosingTags += chunkString;
					leftover = "";
					continue;
				}
				const bodyEndIndex = chunkString.indexOf(BODY_END_TAG);
				const htmlEndIndex = chunkString.indexOf(HTML_END_TAG);
				if (bodyEndIndex !== -1 && htmlEndIndex !== -1 && bodyEndIndex < htmlEndIndex) {
					pendingClosingTags = chunkString.slice(bodyEndIndex);
					safeEnqueue(chunkString.slice(0, bodyEndIndex));
					flushPendingRouterHtml();
					leftover = "";
					continue;
				}
				const lastClosingTagEnd = findLastClosingTagEnd(chunkString);
				if (lastClosingTagEnd > 0) {
					safeEnqueue(chunkString.slice(0, lastClosingTagEnd));
					flushPendingRouterHtml();
					leftover = chunkString.slice(lastClosingTagEnd);
					if (leftover.length > MAX_LEFTOVER_CHARS) {
						safeEnqueue(leftover.slice(0, leftover.length - MAX_LEFTOVER_CHARS));
						leftover = leftover.slice(-MAX_LEFTOVER_CHARS);
					}
				} else {
					const combined = chunkString;
					if (combined.length > MAX_LEFTOVER_CHARS) {
						const flushUpto = combined.length - MAX_LEFTOVER_CHARS;
						safeEnqueue(combined.slice(0, flushUpto));
						leftover = combined.slice(flushUpto);
					} else leftover = combined;
				}
			}
			if (cleanedUp || isStreamClosed) return;
			isAppRendering = false;
			router.serverSsr?.setRenderFinished();
			if (serializationFinished) tryFinish();
			else {
				const timeoutMs = opts?.timeoutMs ?? DEFAULT_SERIALIZATION_TIMEOUT_MS;
				serializationTimeoutHandle = setTimeout(() => {
					if (!cleanedUp && !isStreamClosed) {
						console.error("Serialization timeout after app render finished");
						safeError(/* @__PURE__ */ new Error("Serialization timeout after app render finished"));
						cleanup();
					}
				}, timeoutMs);
			}
		} catch (error) {
			if (cleanedUp) return;
			console.error("Error reading appStream:", error);
			isAppRendering = false;
			router.serverSsr?.setRenderFinished();
			safeError(error);
			cleanup();
		} finally {
			reader.releaseLock();
		}
	})().catch((error) => {
		if (cleanedUp) return;
		console.error("Error in stream transform:", error);
		safeError(error);
		cleanup();
	});
	return stream;
}
//#endregion
exports.transformPipeableStreamWithRouter = transformPipeableStreamWithRouter;
exports.transformReadableStreamWithRouter = transformReadableStreamWithRouter;
exports.transformStreamWithRouter = transformStreamWithRouter;

//# sourceMappingURL=transformStreamWithRouter.cjs.map