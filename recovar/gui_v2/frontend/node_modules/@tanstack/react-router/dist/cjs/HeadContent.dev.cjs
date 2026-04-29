const require_runtime = require("./_virtual/_rolldown/runtime.cjs");
const require_ClientOnly = require("./ClientOnly.cjs");
const require_useRouter = require("./useRouter.cjs");
const require_Asset = require("./Asset.cjs");
const require_headContentUtils = require("./headContentUtils.cjs");
let react = require("react");
react = require_runtime.__toESM(react);
let react_jsx_runtime = require("react/jsx-runtime");
//#region src/HeadContent.dev.tsx
var DEV_STYLES_ATTR = "data-tanstack-router-dev-styles";
/**
* Render route-managed head tags (title, meta, links, styles, head scripts).
* Place inside the document head of your app shell.
*
* Development version: filters out dev styles link after hydration and
* includes a fallback cleanup effect for hydration mismatch cases.
*
* @link https://tanstack.com/router/latest/docs/framework/react/guide/document-head-management
*/
function HeadContent(props) {
	const tags = require_headContentUtils.useTags(props.assetCrossOrigin);
	const nonce = require_useRouter.useRouter().options.ssr?.nonce;
	const hydrated = require_ClientOnly.useHydrated();
	react.useEffect(() => {
		if (hydrated) document.querySelectorAll(`link[${DEV_STYLES_ATTR}]`).forEach((el) => el.remove());
	}, [hydrated]);
	return /* @__PURE__ */ (0, react_jsx_runtime.jsx)(react_jsx_runtime.Fragment, { children: (hydrated ? tags.filter((tag) => !tag.attrs?.[DEV_STYLES_ATTR]) : tags).map((tag) => /* @__PURE__ */ (0, react.createElement)(require_Asset.Asset, {
		...tag,
		key: `tsr-meta-${JSON.stringify(tag)}`,
		nonce
	})) });
}
//#endregion
exports.HeadContent = HeadContent;

//# sourceMappingURL=HeadContent.dev.cjs.map