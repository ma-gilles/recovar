require("./_virtual/_rolldown/runtime.cjs");
const require_useRouter = require("./useRouter.cjs");
let _tanstack_router_core = require("@tanstack/router-core");
//#region src/ScrollRestoration.tsx
function useScrollRestoration() {
	(0, _tanstack_router_core.setupScrollRestoration)(require_useRouter.useRouter(), true);
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
	return (0, _tanstack_router_core.getElementScrollRestorationEntry)(require_useRouter.useRouter(), options);
}
//#endregion
exports.ScrollRestoration = ScrollRestoration;
exports.useElementScrollRestoration = useElementScrollRestoration;

//# sourceMappingURL=ScrollRestoration.cjs.map