require("./_virtual/_rolldown/runtime.cjs");
const require_useRouter = require("./useRouter.cjs");
const require_ScriptOnce = require("./ScriptOnce.cjs");
let react_jsx_runtime = require("react/jsx-runtime");
let _tanstack_router_core_scroll_restoration_script = require("@tanstack/router-core/scroll-restoration-script");
//#region src/scroll-restoration.tsx
function ScrollRestoration() {
	const script = (0, _tanstack_router_core_scroll_restoration_script.getScrollRestorationScriptForRouter)(require_useRouter.useRouter());
	if (!script) return null;
	return /* @__PURE__ */ (0, react_jsx_runtime.jsx)(require_ScriptOnce.ScriptOnce, { children: script });
}
//#endregion
exports.ScrollRestoration = ScrollRestoration;

//# sourceMappingURL=scroll-restoration.cjs.map