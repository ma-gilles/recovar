const require_invariant = require("./invariant.cjs");
const require_path = require("./path.cjs");
const require_not_found = require("./not-found.cjs");
const require_root = require("./root.cjs");
const require_redirect = require("./redirect.cjs");
//#region src/route.ts
var BaseRoute = class {
	get to() {
		return this._to;
	}
	get id() {
		return this._id;
	}
	get path() {
		return this._path;
	}
	get fullPath() {
		return this._fullPath;
	}
	constructor(options) {
		this.init = (opts) => {
			this.originalIndex = opts.originalIndex;
			const options = this.options;
			const isRoot = !options?.path && !options?.id;
			this.parentRoute = this.options.getParentRoute?.();
			if (isRoot) this._path = require_root.rootRouteId;
			else if (!this.parentRoute) {
				if (process.env.NODE_ENV !== "production") throw new Error(`Invariant failed: Child Route instances must pass a 'getParentRoute: () => ParentRoute' option that returns a Route instance.`);
				require_invariant.invariant();
			}
			let path = isRoot ? require_root.rootRouteId : options?.path;
			if (path && path !== "/") path = require_path.trimPathLeft(path);
			const customId = options?.id || path;
			let id = isRoot ? require_root.rootRouteId : require_path.joinPaths([this.parentRoute.id === "__root__" ? "" : this.parentRoute.id, customId]);
			if (path === "__root__") path = "/";
			if (id !== "__root__") id = require_path.joinPaths(["/", id]);
			const fullPath = id === "__root__" ? "/" : require_path.joinPaths([this.parentRoute.fullPath, path]);
			this._path = path;
			this._id = id;
			this._fullPath = fullPath;
			this._to = require_path.trimPathRight(fullPath);
		};
		this.addChildren = (children) => {
			return this._addFileChildren(children);
		};
		this._addFileChildren = (children) => {
			if (Array.isArray(children)) this.children = children;
			if (typeof children === "object" && children !== null) this.children = Object.values(children);
			return this;
		};
		this._addFileTypes = () => {
			return this;
		};
		this.updateLoader = (options) => {
			Object.assign(this.options, options);
			return this;
		};
		this.update = (options) => {
			Object.assign(this.options, options);
			return this;
		};
		this.lazy = (lazyFn) => {
			this.lazyFn = lazyFn;
			return this;
		};
		this.redirect = (opts) => require_redirect.redirect({
			from: this.fullPath,
			...opts
		});
		this.options = options || {};
		this.isRoot = !options?.getParentRoute;
		if (options?.id && options?.path) throw new Error(`Route cannot have both an 'id' and a 'path' option.`);
	}
};
var BaseRouteApi = class {
	constructor({ id }) {
		this.notFound = (opts) => {
			return require_not_found.notFound({
				routeId: this.id,
				...opts
			});
		};
		this.redirect = (opts) => require_redirect.redirect({
			from: this.id,
			...opts
		});
		this.id = id;
	}
};
var BaseRootRoute = class extends BaseRoute {
	constructor(options) {
		super(options);
	}
};
//#endregion
exports.BaseRootRoute = BaseRootRoute;
exports.BaseRoute = BaseRoute;
exports.BaseRouteApi = BaseRouteApi;

//# sourceMappingURL=route.cjs.map