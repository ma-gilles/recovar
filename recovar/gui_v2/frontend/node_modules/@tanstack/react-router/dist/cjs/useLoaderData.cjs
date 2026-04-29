const require_useMatch = require("./useMatch.cjs");
//#region src/useLoaderData.tsx
/**
* Read and select the current route's loader data with type‑safety.
*
* Options:
* - `from`/`strict`: Choose which route's data to read and strictness
* - `select`: Map the loader data to a derived value
* - `structuralSharing`: Enable structural sharing for stable references
*
* @returns The loader data (or selected value) for the matched route.
* @link https://tanstack.com/router/latest/docs/framework/react/api/router/useLoaderDataHook
*/
function useLoaderData(opts) {
	return require_useMatch.useMatch({
		from: opts.from,
		strict: opts.strict,
		structuralSharing: opts.structuralSharing,
		select: (s) => {
			return opts.select ? opts.select(s.loaderData) : s.loaderData;
		}
	});
}
//#endregion
exports.useLoaderData = useLoaderData;

//# sourceMappingURL=useLoaderData.cjs.map