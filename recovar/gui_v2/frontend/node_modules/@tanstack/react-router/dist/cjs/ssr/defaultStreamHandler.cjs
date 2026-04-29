require("../_virtual/_rolldown/runtime.cjs");
const require_RouterServer = require("./RouterServer.cjs");
const require_renderRouterToStream = require("./renderRouterToStream.cjs");
let react_jsx_runtime = require("react/jsx-runtime");
//#region src/ssr/defaultStreamHandler.tsx
var defaultStreamHandler = (0, require("@tanstack/router-core/ssr/server").defineHandlerCallback)(({ request, router, responseHeaders }) => require_renderRouterToStream.renderRouterToStream({
	request,
	router,
	responseHeaders,
	children: /* @__PURE__ */ (0, react_jsx_runtime.jsx)(require_RouterServer.RouterServer, { router })
}));
//#endregion
exports.defaultStreamHandler = defaultStreamHandler;

//# sourceMappingURL=defaultStreamHandler.cjs.map