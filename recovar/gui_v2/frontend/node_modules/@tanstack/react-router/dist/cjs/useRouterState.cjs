require("./_virtual/_rolldown/runtime.cjs");
const require_useRouter = require("./useRouter.cjs");
let _tanstack_router_core = require("@tanstack/router-core");
let react = require("react");
let _tanstack_react_store = require("@tanstack/react-store");
let _tanstack_router_core_isServer = require("@tanstack/router-core/isServer");
//#region src/useRouterState.tsx
/**
* Subscribe to the router's state store with optional selection and
* structural sharing for render optimization.
*
* Options:
* - `select`: Project the full router state to a derived slice
* - `structuralSharing`: Replace-equal semantics for stable references
* - `router`: Read state from a specific router instance instead of context
*
* @returns The selected router state (or the full state by default).
* @link https://tanstack.com/router/latest/docs/framework/react/api/router/useRouterStateHook
*/
function useRouterState(opts) {
	const contextRouter = require_useRouter.useRouter({ warn: opts?.router === void 0 });
	const router = opts?.router || contextRouter;
	if (_tanstack_router_core_isServer.isServer ?? router.isServer) {
		const state = router.stores.__store.state;
		return opts?.select ? opts.select(state) : state;
	}
	const previousResult = (0, react.useRef)(void 0);
	return (0, _tanstack_react_store.useStore)(router.stores.__store, (state) => {
		if (opts?.select) {
			if (opts.structuralSharing ?? router.options.defaultStructuralSharing) {
				const newSlice = (0, _tanstack_router_core.replaceEqualDeep)(previousResult.current, opts.select(state));
				previousResult.current = newSlice;
				return newSlice;
			}
			return opts.select(state);
		}
		return state;
	});
}
//#endregion
exports.useRouterState = useRouterState;

//# sourceMappingURL=useRouterState.cjs.map