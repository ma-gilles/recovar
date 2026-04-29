const require_runtime = require("../_virtual/_rolldown/runtime.cjs");
let _tanstack_router_core_ssr_server = require("@tanstack/router-core/ssr/server");
let react_dom_server = require("react-dom/server");
react_dom_server = require_runtime.__toESM(react_dom_server);
let node_stream = require("node:stream");
let isbot = require("isbot");
//#region src/ssr/renderRouterToStream.tsx
var renderRouterToStream = async ({ request, router, responseHeaders, children }) => {
	if (typeof react_dom_server.default.renderToReadableStream === "function") {
		const stream = await react_dom_server.default.renderToReadableStream(children, {
			signal: request.signal,
			nonce: router.options.ssr?.nonce,
			progressiveChunkSize: Number.POSITIVE_INFINITY
		});
		if ((0, isbot.isbot)(request.headers.get("User-Agent"))) await stream.allReady;
		const responseStream = (0, _tanstack_router_core_ssr_server.transformReadableStreamWithRouter)(router, stream);
		return new Response(responseStream, {
			status: router.stores.statusCode.state,
			headers: responseHeaders
		});
	}
	if (typeof react_dom_server.default.renderToPipeableStream === "function") {
		const reactAppPassthrough = new node_stream.PassThrough();
		try {
			const pipeable = react_dom_server.default.renderToPipeableStream(children, {
				nonce: router.options.ssr?.nonce,
				progressiveChunkSize: Number.POSITIVE_INFINITY,
				...(0, isbot.isbot)(request.headers.get("User-Agent")) ? { onAllReady() {
					pipeable.pipe(reactAppPassthrough);
				} } : { onShellReady() {
					pipeable.pipe(reactAppPassthrough);
				} },
				onError: (error, info) => {
					console.error("Error in renderToPipeableStream:", error, info);
					if (!reactAppPassthrough.destroyed) reactAppPassthrough.destroy(error instanceof Error ? error : new Error(String(error)));
				}
			});
		} catch (e) {
			console.error("Error in renderToPipeableStream:", e);
			reactAppPassthrough.destroy(e instanceof Error ? e : new Error(String(e)));
		}
		const responseStream = (0, _tanstack_router_core_ssr_server.transformPipeableStreamWithRouter)(router, reactAppPassthrough);
		return new Response(responseStream, {
			status: router.stores.statusCode.state,
			headers: responseHeaders
		});
	}
	throw new Error("No renderToReadableStream or renderToPipeableStream found in react-dom/server. Ensure you are using a version of react-dom that supports streaming.");
};
//#endregion
exports.renderRouterToStream = renderRouterToStream;

//# sourceMappingURL=renderRouterToStream.cjs.map