import { useRouter } from "./useRouter.js";
import { useMatch } from "./useMatch.js";
import { useLoaderData } from "./useLoaderData.js";
import { useLoaderDeps } from "./useLoaderDeps.js";
import { useParams } from "./useParams.js";
import { useSearch } from "./useSearch.js";
import { useNavigate } from "./useNavigate.js";
import { useRouteContext } from "./useRouteContext.js";
import { Link } from "./link.js";
import { BaseRootRoute, BaseRoute, BaseRouteApi, notFound } from "@tanstack/router-core";
import React from "react";
import { jsx } from "react/jsx-runtime";
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
var RouteApi = class extends BaseRouteApi {
	/**
	* @deprecated Use the `getRouteApi` function instead.
	*/
	constructor({ id }) {
		super({ id });
		this.useMatch = (opts) => {
			return useMatch({
				select: opts?.select,
				from: this.id,
				structuralSharing: opts?.structuralSharing
			});
		};
		this.useRouteContext = (opts) => {
			return useRouteContext({
				...opts,
				from: this.id
			});
		};
		this.useSearch = (opts) => {
			return useSearch({
				select: opts?.select,
				structuralSharing: opts?.structuralSharing,
				from: this.id
			});
		};
		this.useParams = (opts) => {
			return useParams({
				select: opts?.select,
				structuralSharing: opts?.structuralSharing,
				from: this.id
			});
		};
		this.useLoaderDeps = (opts) => {
			return useLoaderDeps({
				...opts,
				from: this.id,
				strict: false
			});
		};
		this.useLoaderData = (opts) => {
			return useLoaderData({
				...opts,
				from: this.id,
				strict: false
			});
		};
		this.useNavigate = () => {
			return useNavigate({ from: useRouter().routesById[this.id].fullPath });
		};
		this.notFound = (opts) => {
			return notFound({
				routeId: this.id,
				...opts
			});
		};
		this.Link = React.forwardRef((props, ref) => {
			const fullPath = useRouter().routesById[this.id].fullPath;
			return /* @__PURE__ */ jsx(Link, {
				ref,
				from: fullPath,
				...props
			});
		});
	}
};
var Route = class extends BaseRoute {
	/**
	* @deprecated Use the `createRoute` function instead.
	*/
	constructor(options) {
		super(options);
		this.useMatch = (opts) => {
			return useMatch({
				select: opts?.select,
				from: this.id,
				structuralSharing: opts?.structuralSharing
			});
		};
		this.useRouteContext = (opts) => {
			return useRouteContext({
				...opts,
				from: this.id
			});
		};
		this.useSearch = (opts) => {
			return useSearch({
				select: opts?.select,
				structuralSharing: opts?.structuralSharing,
				from: this.id
			});
		};
		this.useParams = (opts) => {
			return useParams({
				select: opts?.select,
				structuralSharing: opts?.structuralSharing,
				from: this.id
			});
		};
		this.useLoaderDeps = (opts) => {
			return useLoaderDeps({
				...opts,
				from: this.id
			});
		};
		this.useLoaderData = (opts) => {
			return useLoaderData({
				...opts,
				from: this.id
			});
		};
		this.useNavigate = () => {
			return useNavigate({ from: this.fullPath });
		};
		this.Link = React.forwardRef((props, ref) => {
			return /* @__PURE__ */ jsx(Link, {
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
var RootRoute = class extends BaseRootRoute {
	/**
	* @deprecated `RootRoute` is now an internal implementation detail. Use `createRootRoute()` instead.
	*/
	constructor(options) {
		super(options);
		this.useMatch = (opts) => {
			return useMatch({
				select: opts?.select,
				from: this.id,
				structuralSharing: opts?.structuralSharing
			});
		};
		this.useRouteContext = (opts) => {
			return useRouteContext({
				...opts,
				from: this.id
			});
		};
		this.useSearch = (opts) => {
			return useSearch({
				select: opts?.select,
				structuralSharing: opts?.structuralSharing,
				from: this.id
			});
		};
		this.useParams = (opts) => {
			return useParams({
				select: opts?.select,
				structuralSharing: opts?.structuralSharing,
				from: this.id
			});
		};
		this.useLoaderDeps = (opts) => {
			return useLoaderDeps({
				...opts,
				from: this.id
			});
		};
		this.useLoaderData = (opts) => {
			return useLoaderData({
				...opts,
				from: this.id
			});
		};
		this.useNavigate = () => {
			return useNavigate({ from: this.fullPath });
		};
		this.Link = React.forwardRef((props, ref) => {
			return /* @__PURE__ */ jsx(Link, {
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
export { NotFoundRoute, RootRoute, Route, RouteApi, createRootRoute, createRootRouteWithContext, createRoute, createRouteMask, getRouteApi, rootRouteWithContext };

//# sourceMappingURL=route.js.map