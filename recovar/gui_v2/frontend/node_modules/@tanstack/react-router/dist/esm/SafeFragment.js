import "react";
import { Fragment, jsx } from "react/jsx-runtime";
//#region src/SafeFragment.tsx
function SafeFragment(props) {
	return /* @__PURE__ */ jsx(Fragment, { children: props.children });
}
//#endregion
export { SafeFragment };

//# sourceMappingURL=SafeFragment.js.map