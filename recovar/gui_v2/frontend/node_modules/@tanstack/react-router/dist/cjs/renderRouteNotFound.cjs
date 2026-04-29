const require_runtime = require("./_virtual/_rolldown/runtime.cjs");
const require_not_found = require("./not-found.cjs");
let react = require("react");
react = require_runtime.__toESM(react);
let react_jsx_runtime = require("react/jsx-runtime");
//#region src/renderRouteNotFound.tsx
/**
* Renders a not found component for a route when no matching route is found.
*
* @param router - The router instance containing the route configuration
* @param route - The route that triggered the not found state
* @param data - Additional data to pass to the not found component
* @returns The rendered not found component or a default fallback component
*/
function renderRouteNotFound(router, route, data) {
	if (!route.options.notFoundComponent) {
		if (router.options.defaultNotFoundComponent) return /* @__PURE__ */ (0, react_jsx_runtime.jsx)(router.options.defaultNotFoundComponent, { ...data });
		if (process.env.NODE_ENV !== "production") {
			if (!route.options.notFoundComponent) console.warn(`Warning: A notFoundError was encountered on the route with ID "${route.id}", but a notFoundComponent option was not configured, nor was a router level defaultNotFoundComponent configured. Consider configuring at least one of these to avoid TanStack Router's overly generic defaultNotFoundComponent (<p>Not Found</p>)`);
		}
		return /* @__PURE__ */ (0, react_jsx_runtime.jsx)(require_not_found.DefaultGlobalNotFound, {});
	}
	return /* @__PURE__ */ (0, react_jsx_runtime.jsx)(route.options.notFoundComponent, { ...data });
}
//#endregion
exports.renderRouteNotFound = renderRouteNotFound;

//# sourceMappingURL=renderRouteNotFound.cjs.map