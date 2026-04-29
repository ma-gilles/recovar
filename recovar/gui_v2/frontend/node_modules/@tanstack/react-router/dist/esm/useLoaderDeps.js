import { useMatch } from "./useMatch.js";
//#region src/useLoaderDeps.tsx
/**
* Read and select the current route's loader dependencies object.
*
* Options:
* - `from`: Choose which route's loader deps to read
* - `select`: Map the deps to a derived value
* - `structuralSharing`: Enable structural sharing for stable references
*
* @returns The loader deps (or selected value) for the matched route.
* @link https://tanstack.com/router/latest/docs/framework/react/api/router/useLoaderDepsHook
*/
function useLoaderDeps(opts) {
	const { select, ...rest } = opts;
	return useMatch({
		...rest,
		select: (s) => {
			return select ? select(s.loaderDeps) : s.loaderDeps;
		}
	});
}
//#endregion
export { useLoaderDeps };

//# sourceMappingURL=useLoaderDeps.js.map