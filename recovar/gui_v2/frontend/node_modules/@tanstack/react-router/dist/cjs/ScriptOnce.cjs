require("./_virtual/_rolldown/runtime.cjs");
const require_useRouter = require("./useRouter.cjs");
let react_jsx_runtime = require("react/jsx-runtime");
let _tanstack_router_core_isServer = require("@tanstack/router-core/isServer");
//#region src/ScriptOnce.tsx
/**
* Server-only helper to emit a script tag exactly once during SSR.
*/
function ScriptOnce({ children }) {
	const router = require_useRouter.useRouter();
	if (!(_tanstack_router_core_isServer.isServer ?? router.isServer)) return null;
	return /* @__PURE__ */ (0, react_jsx_runtime.jsx)("script", {
		nonce: router.options.ssr?.nonce,
		dangerouslySetInnerHTML: { __html: children + ";document.currentScript.remove()" }
	});
}
//#endregion
exports.ScriptOnce = ScriptOnce;

//# sourceMappingURL=ScriptOnce.cjs.map