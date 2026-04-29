import ReactDOMServer from "react-dom/server";
//#region src/ssr/renderRouterToString.tsx
var renderRouterToString = async ({ router, responseHeaders, children }) => {
	try {
		let html = ReactDOMServer.renderToString(children);
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
export { renderRouterToString };

//# sourceMappingURL=renderRouterToString.js.map