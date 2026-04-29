const require_runtime = require("./_virtual/_rolldown/runtime.cjs");
const require_useRouter = require("./useRouter.cjs");
let react = require("react");
react = require_runtime.__toESM(react);
//#region src/useBlocker.tsx
function _resolveBlockerOpts(opts, condition) {
	if (opts === void 0) return {
		shouldBlockFn: () => true,
		withResolver: false
	};
	if ("shouldBlockFn" in opts) return opts;
	if (typeof opts === "function") {
		const shouldBlock = Boolean(condition ?? true);
		const _customBlockerFn = async () => {
			if (shouldBlock) return await opts();
			return false;
		};
		return {
			shouldBlockFn: _customBlockerFn,
			enableBeforeUnload: shouldBlock,
			withResolver: false
		};
	}
	const shouldBlock = Boolean(opts.condition ?? true);
	const fn = opts.blockerFn;
	const _customBlockerFn = async () => {
		if (shouldBlock && fn !== void 0) return await fn();
		return shouldBlock;
	};
	return {
		shouldBlockFn: _customBlockerFn,
		enableBeforeUnload: shouldBlock,
		withResolver: fn === void 0
	};
}
function useBlocker(opts, condition) {
	const { shouldBlockFn, enableBeforeUnload = true, disabled = false, withResolver = false } = _resolveBlockerOpts(opts, condition);
	const router = require_useRouter.useRouter();
	const { history } = router;
	const [resolver, setResolver] = react.useState({
		status: "idle",
		current: void 0,
		next: void 0,
		action: void 0,
		proceed: void 0,
		reset: void 0
	});
	react.useEffect(() => {
		const blockerFnComposed = async (blockerFnArgs) => {
			function getLocation(location) {
				const parsedLocation = router.parseLocation(location);
				const matchedRoutes = router.getMatchedRoutes(parsedLocation.pathname);
				if (matchedRoutes.foundRoute === void 0) return {
					routeId: "__notFound__",
					fullPath: parsedLocation.pathname,
					pathname: parsedLocation.pathname,
					params: matchedRoutes.routeParams,
					search: router.options.parseSearch(location.search)
				};
				return {
					routeId: matchedRoutes.foundRoute.id,
					fullPath: matchedRoutes.foundRoute.fullPath,
					pathname: parsedLocation.pathname,
					params: matchedRoutes.routeParams,
					search: router.options.parseSearch(location.search)
				};
			}
			const current = getLocation(blockerFnArgs.currentLocation);
			const next = getLocation(blockerFnArgs.nextLocation);
			if (current.routeId === "__notFound__" && next.routeId !== "__notFound__") return false;
			const shouldBlock = await shouldBlockFn({
				action: blockerFnArgs.action,
				current,
				next
			});
			if (!withResolver) return shouldBlock;
			if (!shouldBlock) return false;
			const canNavigateAsync = await new Promise((resolve) => {
				setResolver({
					status: "blocked",
					current,
					next,
					action: blockerFnArgs.action,
					proceed: () => resolve(false),
					reset: () => resolve(true)
				});
			});
			setResolver({
				status: "idle",
				current: void 0,
				next: void 0,
				action: void 0,
				proceed: void 0,
				reset: void 0
			});
			return canNavigateAsync;
		};
		return disabled ? void 0 : history.block({
			blockerFn: blockerFnComposed,
			enableBeforeUnload
		});
	}, [
		shouldBlockFn,
		enableBeforeUnload,
		disabled,
		withResolver,
		history,
		router
	]);
	return resolver;
}
var _resolvePromptBlockerArgs = (props) => {
	if ("shouldBlockFn" in props) return { ...props };
	const shouldBlock = Boolean(props.condition ?? true);
	const fn = props.blockerFn;
	const _customBlockerFn = async () => {
		if (shouldBlock && fn !== void 0) return await fn();
		return shouldBlock;
	};
	return {
		shouldBlockFn: _customBlockerFn,
		enableBeforeUnload: shouldBlock,
		withResolver: fn === void 0
	};
};
function Block(opts) {
	const { children, ...rest } = opts;
	const resolver = useBlocker(_resolvePromptBlockerArgs(rest));
	return children ? typeof children === "function" ? children(resolver) : children : null;
}
//#endregion
exports.Block = Block;
exports.useBlocker = useBlocker;

//# sourceMappingURL=useBlocker.cjs.map