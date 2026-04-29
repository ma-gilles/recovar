const require_runtime = require("./_virtual/_rolldown/runtime.cjs");
const require_routerContext = require("./routerContext.cjs");
const require_Matches = require("./Matches.cjs");
let react = require("react");
react = require_runtime.__toESM(react);
let react_jsx_runtime = require("react/jsx-runtime");
//#region src/RouterProvider.tsx
/**
* Low-level provider that places the router into React context and optionally
* updates router options from props. Most apps should use `RouterProvider`.
*/
function RouterContextProvider({ router, children, ...rest }) {
	if (Object.keys(rest).length > 0) router.update({
		...router.options,
		...rest,
		context: {
			...router.options.context,
			...rest.context
		}
	});
	const provider = /* @__PURE__ */ (0, react_jsx_runtime.jsx)(require_routerContext.routerContext.Provider, {
		value: router,
		children
	});
	if (router.options.Wrap) return /* @__PURE__ */ (0, react_jsx_runtime.jsx)(router.options.Wrap, { children: provider });
	return provider;
}
/**
* Top-level component that renders the active route matches and provides the
* router to the React tree via context.
*
* Accepts the same options as `createRouter` via props to update the router
* instance after creation.
*
* @link https://tanstack.com/router/latest/docs/framework/react/api/router/createRouterFunction
*/
function RouterProvider({ router, ...rest }) {
	return /* @__PURE__ */ (0, react_jsx_runtime.jsx)(RouterContextProvider, {
		router,
		...rest,
		children: /* @__PURE__ */ (0, react_jsx_runtime.jsx)(require_Matches.Matches, {})
	});
}
//#endregion
exports.RouterContextProvider = RouterContextProvider;
exports.RouterProvider = RouterProvider;

//# sourceMappingURL=RouterProvider.cjs.map