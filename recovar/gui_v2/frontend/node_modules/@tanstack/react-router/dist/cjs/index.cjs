Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
require("./_virtual/_rolldown/runtime.cjs");
const require_awaited = require("./awaited.cjs");
const require_CatchBoundary = require("./CatchBoundary.cjs");
const require_ClientOnly = require("./ClientOnly.cjs");
const require_useRouter = require("./useRouter.cjs");
const require_useMatch = require("./useMatch.cjs");
const require_useLoaderData = require("./useLoaderData.cjs");
const require_useLoaderDeps = require("./useLoaderDeps.cjs");
const require_useParams = require("./useParams.cjs");
const require_useSearch = require("./useSearch.cjs");
const require_useNavigate = require("./useNavigate.cjs");
const require_useRouteContext = require("./useRouteContext.cjs");
const require_link = require("./link.cjs");
const require_route = require("./route.cjs");
const require_fileRoute = require("./fileRoute.cjs");
const require_lazyRouteComponent = require("./lazyRouteComponent.cjs");
const require_not_found = require("./not-found.cjs");
const require_ScriptOnce = require("./ScriptOnce.cjs");
const require_Match = require("./Match.cjs");
const require_Matches = require("./Matches.cjs");
const require_router = require("./router.cjs");
const require_RouterProvider = require("./RouterProvider.cjs");
const require_ScrollRestoration = require("./ScrollRestoration.cjs");
const require_useBlocker = require("./useBlocker.cjs");
const require_useRouterState = require("./useRouterState.cjs");
const require_useLocation = require("./useLocation.cjs");
const require_useCanGoBack = require("./useCanGoBack.cjs");
const require_Asset = require("./Asset.cjs");
const require_headContentUtils = require("./headContentUtils.cjs");
const require_HeadContent = require("./HeadContent.cjs");
const require_Scripts = require("./Scripts.cjs");
let _tanstack_router_core = require("@tanstack/router-core");
let _tanstack_history = require("@tanstack/history");
exports.Asset = require_Asset.Asset;
exports.Await = require_awaited.Await;
exports.Block = require_useBlocker.Block;
exports.CatchBoundary = require_CatchBoundary.CatchBoundary;
exports.CatchNotFound = require_not_found.CatchNotFound;
exports.ClientOnly = require_ClientOnly.ClientOnly;
Object.defineProperty(exports, "DEFAULT_PROTOCOL_ALLOWLIST", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.DEFAULT_PROTOCOL_ALLOWLIST;
	}
});
exports.DefaultGlobalNotFound = require_not_found.DefaultGlobalNotFound;
exports.ErrorComponent = require_CatchBoundary.ErrorComponent;
exports.FileRoute = require_fileRoute.FileRoute;
exports.FileRouteLoader = require_fileRoute.FileRouteLoader;
exports.HeadContent = require_HeadContent.HeadContent;
exports.LazyRoute = require_fileRoute.LazyRoute;
exports.Link = require_link.Link;
exports.Match = require_Match.Match;
exports.MatchRoute = require_Matches.MatchRoute;
exports.Matches = require_Matches.Matches;
exports.Navigate = require_useNavigate.Navigate;
exports.NotFoundRoute = require_route.NotFoundRoute;
exports.Outlet = require_Match.Outlet;
exports.RootRoute = require_route.RootRoute;
exports.Route = require_route.Route;
exports.RouteApi = require_route.RouteApi;
exports.Router = require_router.Router;
exports.RouterContextProvider = require_RouterProvider.RouterContextProvider;
exports.RouterProvider = require_RouterProvider.RouterProvider;
exports.ScriptOnce = require_ScriptOnce.ScriptOnce;
exports.Scripts = require_Scripts.Scripts;
exports.ScrollRestoration = require_ScrollRestoration.ScrollRestoration;
Object.defineProperty(exports, "SearchParamError", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.SearchParamError;
	}
});
Object.defineProperty(exports, "cleanPath", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.cleanPath;
	}
});
Object.defineProperty(exports, "composeRewrites", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.composeRewrites;
	}
});
Object.defineProperty(exports, "createBrowserHistory", {
	enumerable: true,
	get: function() {
		return _tanstack_history.createBrowserHistory;
	}
});
Object.defineProperty(exports, "createControlledPromise", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.createControlledPromise;
	}
});
exports.createFileRoute = require_fileRoute.createFileRoute;
Object.defineProperty(exports, "createHashHistory", {
	enumerable: true,
	get: function() {
		return _tanstack_history.createHashHistory;
	}
});
Object.defineProperty(exports, "createHistory", {
	enumerable: true,
	get: function() {
		return _tanstack_history.createHistory;
	}
});
exports.createLazyFileRoute = require_fileRoute.createLazyFileRoute;
exports.createLazyRoute = require_fileRoute.createLazyRoute;
exports.createLink = require_link.createLink;
Object.defineProperty(exports, "createMemoryHistory", {
	enumerable: true,
	get: function() {
		return _tanstack_history.createMemoryHistory;
	}
});
exports.createRootRoute = require_route.createRootRoute;
exports.createRootRouteWithContext = require_route.createRootRouteWithContext;
exports.createRoute = require_route.createRoute;
exports.createRouteMask = require_route.createRouteMask;
exports.createRouter = require_router.createRouter;
Object.defineProperty(exports, "createRouterConfig", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.createRouterConfig;
	}
});
Object.defineProperty(exports, "createSerializationAdapter", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.createSerializationAdapter;
	}
});
Object.defineProperty(exports, "deepEqual", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.deepEqual;
	}
});
Object.defineProperty(exports, "defaultParseSearch", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.defaultParseSearch;
	}
});
Object.defineProperty(exports, "defaultStringifySearch", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.defaultStringifySearch;
	}
});
Object.defineProperty(exports, "defer", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.defer;
	}
});
Object.defineProperty(exports, "functionalUpdate", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.functionalUpdate;
	}
});
exports.getRouteApi = require_route.getRouteApi;
Object.defineProperty(exports, "interpolatePath", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.interpolatePath;
	}
});
Object.defineProperty(exports, "isMatch", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.isMatch;
	}
});
Object.defineProperty(exports, "isNotFound", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.isNotFound;
	}
});
Object.defineProperty(exports, "isPlainArray", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.isPlainArray;
	}
});
Object.defineProperty(exports, "isPlainObject", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.isPlainObject;
	}
});
Object.defineProperty(exports, "isRedirect", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.isRedirect;
	}
});
Object.defineProperty(exports, "joinPaths", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.joinPaths;
	}
});
Object.defineProperty(exports, "lazyFn", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.lazyFn;
	}
});
exports.lazyRouteComponent = require_lazyRouteComponent.lazyRouteComponent;
exports.linkOptions = require_link.linkOptions;
Object.defineProperty(exports, "notFound", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.notFound;
	}
});
Object.defineProperty(exports, "parseSearchWith", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.parseSearchWith;
	}
});
Object.defineProperty(exports, "redirect", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.redirect;
	}
});
Object.defineProperty(exports, "replaceEqualDeep", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.replaceEqualDeep;
	}
});
Object.defineProperty(exports, "resolvePath", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.resolvePath;
	}
});
Object.defineProperty(exports, "retainSearchParams", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.retainSearchParams;
	}
});
Object.defineProperty(exports, "rootRouteId", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.rootRouteId;
	}
});
exports.rootRouteWithContext = require_route.rootRouteWithContext;
Object.defineProperty(exports, "stringifySearchWith", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.stringifySearchWith;
	}
});
Object.defineProperty(exports, "stripSearchParams", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.stripSearchParams;
	}
});
Object.defineProperty(exports, "trimPath", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.trimPath;
	}
});
Object.defineProperty(exports, "trimPathLeft", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.trimPathLeft;
	}
});
Object.defineProperty(exports, "trimPathRight", {
	enumerable: true,
	get: function() {
		return _tanstack_router_core.trimPathRight;
	}
});
exports.useAwaited = require_awaited.useAwaited;
exports.useBlocker = require_useBlocker.useBlocker;
exports.useCanGoBack = require_useCanGoBack.useCanGoBack;
exports.useChildMatches = require_Matches.useChildMatches;
exports.useElementScrollRestoration = require_ScrollRestoration.useElementScrollRestoration;
exports.useHydrated = require_ClientOnly.useHydrated;
exports.useLinkProps = require_link.useLinkProps;
exports.useLoaderData = require_useLoaderData.useLoaderData;
exports.useLoaderDeps = require_useLoaderDeps.useLoaderDeps;
exports.useLocation = require_useLocation.useLocation;
exports.useMatch = require_useMatch.useMatch;
exports.useMatchRoute = require_Matches.useMatchRoute;
exports.useMatches = require_Matches.useMatches;
exports.useNavigate = require_useNavigate.useNavigate;
exports.useParams = require_useParams.useParams;
exports.useParentMatches = require_Matches.useParentMatches;
exports.useRouteContext = require_useRouteContext.useRouteContext;
exports.useRouter = require_useRouter.useRouter;
exports.useRouterState = require_useRouterState.useRouterState;
exports.useSearch = require_useSearch.useSearch;
exports.useTags = require_headContentUtils.useTags;
