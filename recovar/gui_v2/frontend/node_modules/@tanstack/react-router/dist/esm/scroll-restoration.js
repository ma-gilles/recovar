import { useRouter } from "./useRouter.js";
import { ScriptOnce } from "./ScriptOnce.js";
import { jsx } from "react/jsx-runtime";
import { getScrollRestorationScriptForRouter } from "@tanstack/router-core/scroll-restoration-script";
//#region src/scroll-restoration.tsx
function ScrollRestoration() {
	const script = getScrollRestorationScriptForRouter(useRouter());
	if (!script) return null;
	return /* @__PURE__ */ jsx(ScriptOnce, { children: script });
}
//#endregion
export { ScrollRestoration };

//# sourceMappingURL=scroll-restoration.js.map