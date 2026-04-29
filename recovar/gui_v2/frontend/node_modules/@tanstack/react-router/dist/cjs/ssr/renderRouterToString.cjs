const require_runtime = require("../_virtual/_rolldown/runtime.cjs");
let react_dom_server = require("react-dom/server");
react_dom_server = require_runtime.__toESM(react_dom_server);
//#region src/ssr/renderRouterToString.tsx
var renderRouterToString = async ({ router, responseHeaders, children }) => {
	try {
		let html = react_dom_server.default.renderToString(children);
		router.serverSsr.setRenderFinished();
		const injectedHtml = router.serverSsr.takeBufferedHtml();
		if (injectedHtml) html = html.replace(`</body>`, () => `${injectedHtml}</body>`);
		return new Response(`<!DOCTYPE html>${html}`, {
			status: router.stores.statusCode.state,
			headers: responseHeaders
		});
	} catch (error) {
		console.error("Render to string error:", error);
		return new Response("Internal Server Error", {
			status: 500,
			headers: responseHeaders
		});
	} finally {
		router.serverSsr?.cleanup();
	}
};
//#endregion
exports.renderRouterToString = renderRouterToString;

//# sourceMappingURL=renderRouterToString.cjs.map