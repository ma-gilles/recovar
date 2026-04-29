import { useRouter } from "./useRouter.js";
import { replaceEqualDeep } from "@tanstack/router-core";
import { useRef } from "react";
import { useStore } from "@tanstack/react-store";
import { isServer } from "@tanstack/router-core/isServer";
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
	const contextRouter = useRouter({ warn: opts?.router === void 0 });
	const router = opts?.router || contextRouter;
	if (isServer ?? router.isServer) {
		const state = router.stores.__store.state;
		return opts?.select ? opts.select(state) : state;
	}
	const previousResult = useRef(void 0);
	return useStore(router.stores.__store, (state) => {
		if (opts?.select) {
			if (opts.structuralSharing ?? router.options.defaultStructuralSharing) {
				const newSlice = replaceEqualDeep(previousResult.current, opts.select(state));
				previousResult.current = newSlice;
				return newSlice;
			}
			return opts.select(state);
		}
		return state;
	});
}
//#endregion
export { useRouterState };

//# sourceMappingURL=useRouterState.js.map