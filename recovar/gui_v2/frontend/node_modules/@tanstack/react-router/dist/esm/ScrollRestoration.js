import { useRouter } from "./useRouter.js";
import { getElementScrollRestorationEntry, setupScrollRestoration } from "@tanstack/router-core";
//#region src/ScrollRestoration.tsx
function useScrollRestoration() {
	setupScrollRestoration(useRouter(), true);
}
/**
* @deprecated Use the `scrollRestoration` router option instead.
*/
function ScrollRestoration(_props) {
	useScrollRestoration();
	if (process.env.NODE_ENV === "development") console.warn("The ScrollRestoration component is deprecated. Use createRouter's `scrollRestoration` option instead.");
	return null;
}
function useElementScrollRestoration(options) {
	useScrollRestoration();
	return getElementScrollRestorationEntry(useRouter(), options);
}
//#endregion
export { ScrollRestoration, useElementScrollRestoration };

//# sourceMappingURL=ScrollRestoration.js.map