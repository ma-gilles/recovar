const require_runtime = require("./_virtual/_rolldown/runtime.cjs");
let react = require("react");
react = require_runtime.__toESM(react);
let react_jsx_runtime = require("react/jsx-runtime");
//#region src/SafeFragment.tsx
function SafeFragment(props) {
	return /* @__PURE__ */ (0, react_jsx_runtime.jsx)(react_jsx_runtime.Fragment, { children: props.children });
}
//#endregion
exports.SafeFragment = SafeFragment;

//# sourceMappingURL=SafeFragment.cjs.map