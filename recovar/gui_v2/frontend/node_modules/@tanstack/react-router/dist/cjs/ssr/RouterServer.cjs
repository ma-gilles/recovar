const require_runtime = require("../_virtual/_rolldown/runtime.cjs");
const require_RouterProvider = require("../RouterProvider.cjs");
let react = require("react");
react = require_runtime.__toESM(react);
let react_jsx_runtime = require("react/jsx-runtime");
//#region src/ssr/RouterServer.tsx
function RouterServer(props) {
	return /* @__PURE__ */ (0, react_jsx_runtime.jsx)(require_RouterProvider.RouterProvider, { router: props.router });
}
//#endregion
exports.RouterServer = RouterServer;

//# sourceMappingURL=RouterServer.cjs.map