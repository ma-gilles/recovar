import { escapeHtml } from "../utils.js";
import { defaultGetScrollRestorationKey, storageKey } from "../scroll-restoration.js";
import scroll_restoration_inline_default from "../scroll-restoration-inline.js";
//#region src/scroll-restoration-script/server.ts
var defaultInlineScrollRestorationScript = `(${scroll_restoration_inline_default})(${escapeHtml(JSON.stringify({
	storageKey,
	shouldScrollRestoration: true
}))})`;
function getScrollRestorationScript(options) {
	if (options.storageKey === "tsr-scroll-restoration-v1_3" && options.shouldScrollRestoration === true && options.key === void 0 && options.behavior === void 0) return defaultInlineScrollRestorationScript;
	return `(${scroll_restoration_inline_default})(${escapeHtml(JSON.stringify(options))})`;
}
function getScrollRestorationScriptForRouter(router) {
	if (typeof router.options.scrollRestoration === "function" && !router.options.scrollRestoration({ location: router.latestLocation })) return null;
	const getKey = router.options.getScrollRestorationKey;
	if (!getKey) return defaultInlineScrollRestorationScript;
	const location = router.latestLocation;
	const userKey = getKey(location);
	if (userKey === defaultGetScrollRestorationKey(location)) return defaultInlineScrollRestorationScript;
	return getScrollRestorationScript({
		storageKey,
		shouldScrollRestoration: true,
		key: userKey
	});
}
//#endregion
export { getScrollRestorationScriptForRouter };

//# sourceMappingURL=server.js.map