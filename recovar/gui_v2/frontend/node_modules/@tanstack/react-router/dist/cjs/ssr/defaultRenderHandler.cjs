require("../_virtual/_rolldown/runtime.cjs");
const require_RouterServer = require("./RouterServer.cjs");
const require_renderRouterToString = require("./renderRouterToString.cjs");
let react_jsx_runtime = require("react/jsx-runtime");
//#region src/ssr/defaultRenderHandler.tsx
var defaultRenderHandler = (0, require("@tanstack/router-core/ssr/server").defineHandlerCallback)(({ router, responseHeaders }) => require_renderRouterToString.renderRouterToString({
	router,
	responseHeaders,
	children: /* @__PURE__ */ (0, react_jsx_runtime.jsx)(require_RouterServer.RouterServer, { router })
}));
//#endregion
exports.defaultRenderHandler = defaultRenderHandler;

//# sourceMappingURL=defaultRenderHandler.cjs.map