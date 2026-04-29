import { useRouter } from "./useRouter.js";
import { Asset } from "./Asset.js";
import { useTags } from "./headContentUtils.js";
import { createElement } from "react";
import { Fragment, jsx } from "react/jsx-runtime";
//#region src/HeadContent.tsx
/**
* Render route-managed head tags (title, meta, links, styles, head scripts).
* Place inside the document head of your app shell.
* @link https://tanstack.com/router/latest/docs/framework/react/guide/document-head-management
*/
function HeadContent(props) {
	const tags = useTags(props.assetCrossOrigin);
	const nonce = useRouter().options.ssr?.nonce;
	return /* @__PURE__ */ jsx(Fragment, { children: tags.map((tag) => /* @__PURE__ */ createElement(Asset, {
		...tag,
		key: `tsr-meta-${JSON.stringify(tag)}`,
		nonce
	})) });
}
//#endregion
export { HeadContent };

//# sourceMappingURL=HeadContent.js.map