import { useMatch } from "./useMatch.js";
//#region src/useRouteContext.ts
function useRouteContext(opts) {
	return useMatch({
		...opts,
		select: (match) => opts.select ? opts.select(match.context) : match.context
	});
}
//#endregion
export { useRouteContext };

//# sourceMappingURL=useRouteContext.js.map