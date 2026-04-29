const require_runtime = require("./_virtual/_rolldown/runtime.cjs");
const require_useRouter = require("./useRouter.cjs");
const require_useMatch = require("./useMatch.cjs");
const require_useLoaderData = require("./useLoaderData.cjs");
const require_useLoaderDeps = require("./useLoaderDeps.cjs");
const require_useParams = require("./useParams.cjs");
const require_useSearch = require("./useSearch.cjs");
const require_useNavigate = require("./useNavigate.cjs");
const require_useRouteContext = require("./useRouteContext.cjs");
const require_link = require("./link.cjs");
let _tanstack_router_core = require("@tanstack/router-core");
let react = require("react");
react = require_runtime.__toESM(react);
let react_jsx_runtime = require("react/jsx-runtime");
//#region src/route.tsx
/**
* Returns a route-specific API that exposes type-safe hooks pre-bound
* to a single route ID. Useful for consuming a route's APIs from files
* where the route object isn't directly imported (e.g. code-split files).
*
* @param id Route ID string literal for the target route.
* @returns A `RouteApi` instance bound to the given route ID.
* @link https://tanstack.com/router/latest/docs/framework/react/api/router/getRouteApiFunction
*/
function getRouteApi(id) {
	return new RouteApi({ id });
}
var RouteApi = class extends _tanstack_router_core.BaseRouteApi {
	/**
	* @deprecated Use the `getRouteApi` function instead.
	*/
	constructor({ id }) {
		super({ id });
		this.useMatch = (opts) => {
			return require_useMatch.useMatch({
				select: opts?.select,
				from: this.id,
				structuralSharing: opts?.structuralSharing
			});
		};
		this.useRouteContext = (opts) => {
			return require_useRouteContext.useRouteContext({
				...opts,
				from: this.id
			});
		};
		this.useSearch = (opts) => {
			return require_useSearch.useSearch({
				select: opts?.select,
				structuralSharing: opts?.structuralSharing,
				from: this.id
			});
		};
		this.useParams = (opts) => {
			return require_useParams.useParams({
				select: opts?.select,
				structuralSharing: opts?.structuralSharing,
				from: this.id
			});
		};
		this.useLoaderDeps = (opts) => {
			return require_useLoaderDeps.useLoaderDeps({
				...opts,
				from: this.id,
				strict: false
			});
		};
		this.useLoaderData = (opts) => {
			return require_useLoaderData.useLoaderData({
				...opts,
				from: this.id,
				strict: false
			});
		};
		this.useNavigate = () => {
			return require_useNavigate.useNavigate({ from: require_useRouter.useRouter().routesById[this.id].fullPath });
		};
		this.notFound = (opts) => {
			return (0, _tanstack_router_core.notFound)({
				routeId: this.id,
				...opts
			});
		};
		this.Link = react.default.forwardRef((props, ref) => {
			const fullPath = require_useRouter.useRouter().routesById[this.id].fullPath;
			return /* @__PURE__ */ (0, react_jsx_runtime.jsx)(require_link.Link, {
				ref,
				from: fullPath,
				...props
			});
		});
	}
};
var Route = class extends _tanstack_router_core.BaseRoute {
	/**
	* @deprecated Use the `createRoute` function instead.
	*/
	constructor(options) {
		super(options);
		this.useMatch = (opts) => {
			return require_useMatch.useMatch({
				select: opts?.select,
				from: this.id,
				structuralSharing: opts?.structuralSharing
			});
		};
		this.useRouteContext = (opts) => {
			return require_useRouteContext.useRouteContext({
				...opts,
				from: this.id
			});
		};
		this.useSearch = (opts) => {
			return require_useSearch.useSearch({
				select: opts?.select,
				structuralSharing: opts?.structuralSharing,
				from: this.id
			});
		};
		this.useParams = (opts) => {
			return require_useParams.useParams({
				select: opts?.select,
				structuralSharing: opts?.structuralSharing,
				from: this.id
			});
		};
		this.useLoaderDeps = (opts) => {
			return require_useLoaderDeps.useLoaderDeps({
				...opts,
				from: this.id
			});
		};
		this.useLoaderData = (opts) => {
			return require_useLoaderData.useLoaderData({
				...opts,
				from: this.id
			});
		};
		this.useNavigate = () => {
			return require_useNavigate.useNavigate({ from: this.fullPath });
		};
		this.Link = react.default.forwardRef((props, ref) => {
			return /* @__PURE__ */ (0, react_jsx_runtime.jsx)(require_link.Link, {
				ref,
				from: this.fullPath,
				...props
			});
		});
	}
};
/**
* Creates a non-root Route instance for code-based routing.
*
* Use this to define a route that will be composed into a route tree
* (typically via a parent route's `addChildren`). If you're using file-based
* routing, prefer `createFileRoute`.
*
* @param options Route options (path, component, loader, context, etc.).
* @returns A Route instance to be attached to the route tree.
* @link https://tanstack.com/router/latest/docs/framework/react/api/router/createRouteFunction
*/
function createRoute(options) {
	return new Route(options);
}
/**
* Creates a root route factory that requires a router context type.
*
* Use when your root route expects `context` to be provided to `createRouter`.
* The returned function behaves like `createRootRoute` but enforces a context type.
*
* @returns A factory function to configure and return a root route.
* @link https://tanstack.com/router/latest/docs/framework/react/api/router/createRootRouteWithContextFunction
*/
function createRootRouteWithContext() {
	return (options) => {
		return createRootRoute(options);
	};
}
/**
* @deprecated Use the `createRootRouteWithContext` function instead.
*/
var rootRouteWithContext = createRootRouteWithContext;
var RootRoute = class extends _tanstack_router_core.BaseRootRoute {
	/**
	* @deprecated `RootRoute` is now an internal implementation detail. Use `createRootRoute()` instead.
	*/
	constructor(options) {
		super(options);
		this.useMatch = (opts) => {
			return require_useMatch.useMatch({
				select: opts?.select,
				from: this.id,
				structuralSharing: opts?.structuralSharing
			});
		};
		this.useRouteContext = (opts) => {
			return require_useRouteContext.useRouteContext({
				...opts,
				from: this.id
			});
		};
		this.useSearch = (opts) => {
			return require_useSearch.useSearch({
				select: opts?.select,
				structuralSharing: opts?.structuralSharing,
				from: this.id
			});
		};
		this.useParams = (opts) => {
			return require_useParams.useParams({
				select: opts?.select,
				structuralSharing: opts?.structuralSharing,
				from: this.id
			});
		};
		this.useLoaderDeps = (opts) => {
			return require_useLoaderDeps.useLoaderDeps({
				...opts,
				from: this.id
			});
		};
		this.useLoaderData = (opts) => {
			return require_useLoaderData.useLoaderData({
				...opts,
				from: this.id
			});
		};
		this.useNavigate = () => {
			return require_useNavigate.useNavigate({ from: this.fullPath });
		};
		this.Link = react.default.forwardRef((props, ref) => {
			return /* @__PURE__ */ (0, react_jsx_runtime.jsx)(require_link.Link, {
				ref,
				from: this.fullPath,
				...props
			});
		});
	}
};
/**
* Creates a root Route instance used to build your route tree.
*
* Typically paired with `createRouter({ routeTree })`. If you need to require
* a typed router context, use `createRootRouteWithContext` instead.
*
* @param options Root route options (component, error, pending, etc.).
* @returns A root route instance.
* @link https://tanstack.com/router/latest/docs/framework/react/api/router/createRootRouteFunction
*/
function createRootRoute(options) {
	return new RootRoute(options);
}
function createRouteMask(opts) {
	return opts;
}
var NotFoundRoute = class extends Route {
	constructor(options) {
		super({
			...options,
			id: "404"
		});
	}
};
//#endregion
exports.NotFoundRoute = NotFoundRoute;
exports.RootRoute = RootRoute;
exports.Route = Route;
exports.RouteApi = RouteApi;
exports.createRootRoute = createRootRoute;
exports.createRootRouteWithContext = createRootRouteWithContext;
exports.createRoute = createRoute;
exports.createRouteMask = createRouteMask;
exports.getRouteApi = getRouteApi;
exports.rootRouteWithContext = rootRouteWithContext;

//# sourceMappingURL=route.cjs.map