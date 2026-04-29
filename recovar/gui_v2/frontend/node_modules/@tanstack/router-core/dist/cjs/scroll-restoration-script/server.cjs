Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
const require_utils = require("../utils.cjs");
const require_scroll_restoration = require("../scroll-restoration.cjs");
const require_scroll_restoration_inline = require("../scroll-restoration-inline.cjs");
//#region src/scroll-restoration-script/server.ts
var defaultInlineScrollRestorationScript = `(${require_scroll_restoration_inline.default})(${require_utils.escapeHtml(JSON.stringify({
	storageKey: require_scroll_restoration.storageKey,
	shouldScrollRestoration: true
}))})`;
function getScrollRestorationScript(options) {
	if (options.storageKey === "tsr-scroll-restoration-v1_3" && options.shouldScrollRestoration === true && options.key === void 0 && options.behavior === void 0) return defaultInlineScrollRestorationScript;
	return `(${require_scroll_restoration_inline.default})(${require_utils.escapeHtml(JSON.stringify(options))})`;
}
function getScrollRestorationScriptForRouter(router) {
	if (typeof router.options.scrollRestoration === "function" && !router.options.scrollRestoration({ location: router.latestLocation })) return null;
	const getKey = router.options.getScrollRestorationKey;
	if (!getKey) return defaultInlineScrollRestorationScript;
	const location = router.latestLocation;
	const userKey = getKey(location);
	if (userKey === require_scroll_restoration.defaultGetScrollRestorationKey(location)) return defaultInlineScrollRestorationScript;
	return getScrollRestorationScript({
		storageKey: require_scroll_restoration.storageKey,
		shouldScrollRestoration: true,
		key: userKey
	});
}
//#endregion
exports.getScrollRestorationScriptForRouter = getScrollRestorationScriptForRouter;

//# sourceMappingURL=server.cjs.map