import { transformPipeableStreamWithRouter, transformReadableStreamWithRouter } from "@tanstack/router-core/ssr/server";
import ReactDOMServer from "react-dom/server";
import { PassThrough } from "node:stream";
import { isbot } from "isbot";
//#region src/ssr/renderRouterToStream.tsx
var renderRouterToStream = async ({ request, router, responseHeaders, children }) => {
	if (typeof ReactDOMServer.renderToReadableStream === "function") {
		const stream = await ReactDOMServer.renderToReadableStream(children, {
			signal: request.signal,
			nonce: router.options.ssr?.nonce,
			progressiveChunkSize: Number.POSITIVE_INFINITY
		});
		if (isbot(request.headers.get("User-Agent"))) await stream.allReady;
		const responseStream = transformReadableStreamWithRouter(router, stream);
		return new Response(responseStream, {
			status: router.stores.statusCode.state,
			headers: responseHeaders
		});
	}
	if (typeof ReactDOMServer.renderToPipeableStream === "function") {
		const reactAppPassthrough = new PassThrough();
		try {
			const pipeable = ReactDOMServer.renderToPipeableStream(children, {
				nonce: router.options.ssr?.nonce,
				progressiveChunkSize: Number.POSITIVE_INFINITY,
				...isbot(request.headers.get("User-Agent")) ? { onAllReady() {
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
		const responseStream = transformPipeableStreamWithRouter(router, reactAppPassthrough);
		return new Response(responseStream, {
			status: router.stores.statusCode.state,
			headers: responseHeaders
		});
	}
	throw new Error("No renderToReadableStream or renderToPipeableStream found in react-dom/server. Ensure you are using a version of react-dom that supports streaming.");
};
//#endregion
export { renderRouterToStream };

//# sourceMappingURL=renderRouterToStream.js.map