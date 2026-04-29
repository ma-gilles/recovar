import { routerContext } from "./routerContext.js";
import { Matches } from "./Matches.js";
import "react";
import { jsx } from "react/jsx-runtime";
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
	const provider = /* @__PURE__ */ jsx(routerContext.Provider, {
		value: router,
		children
	});
	if (router.options.Wrap) return /* @__PURE__ */ jsx(router.options.Wrap, { children: provider });
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
	return /* @__PURE__ */ jsx(RouterContextProvider, {
		router,
		...rest,
		children: /* @__PURE__ */ jsx(Matches, {})
	});
}
//#endregion
export { RouterContextProvider, RouterProvider };

//# sourceMappingURL=RouterProvider.js.map