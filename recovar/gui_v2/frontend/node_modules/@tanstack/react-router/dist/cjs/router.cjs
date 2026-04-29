require("./_virtual/_rolldown/runtime.cjs");
const require_routerStores = require("./routerStores.cjs");
let _tanstack_router_core = require("@tanstack/router-core");
//#region src/router.ts
/**
* Creates a new Router instance for React.
*
* Pass the returned router to `RouterProvider` to enable routing.
* Notable options: `routeTree` (your route definitions) and `context`
* (required if the root route was created with `createRootRouteWithContext`).
*
* @param options Router options used to configure the router.
* @returns A Router instance to be provided to `RouterProvider`.
* @link https://tanstack.com/router/latest/docs/framework/react/api/router/createRouterFunction
*/
var createRouter = (options) => {
	return new Router(options);
};
var Router = class extends _tanstack_router_core.RouterCore {
	constructor(options) {
		super(options, require_routerStores.getStoreFactory);
	}
};
//#endregion
exports.Router = Router;
exports.createRouter = createRouter;

//# sourceMappingURL=router.cjs.map