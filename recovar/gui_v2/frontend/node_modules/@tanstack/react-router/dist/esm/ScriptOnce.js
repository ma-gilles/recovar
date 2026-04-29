import { useRouter } from "./useRouter.js";
import { jsx } from "react/jsx-runtime";
import { isServer } from "@tanstack/router-core/isServer";
//#region src/ScriptOnce.tsx
/**
* Server-only helper to emit a script tag exactly once during SSR.
*/
function ScriptOnce({ children }) {
	const router = useRouter();
	if (!(isServer ?? router.isServer)) return null;
	return /* @__PURE__ */ jsx("script", {
		nonce: router.options.ssr?.nonce,
		dangerouslySetInnerHTML: { __html: children + ";document.currentScript.remove()" }
	});
}
//#endregion
export { ScriptOnce };

//# sourceMappingURL=ScriptOnce.js.map