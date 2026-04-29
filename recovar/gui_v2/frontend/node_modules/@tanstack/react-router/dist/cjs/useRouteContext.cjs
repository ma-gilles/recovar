const require_useMatch = require("./useMatch.cjs");
//#region src/useRouteContext.ts
function useRouteContext(opts) {
	return require_useMatch.useMatch({
		...opts,
		select: (match) => opts.select ? opts.select(match.context) : match.context
	});
}
//#endregion
exports.useRouteContext = useRouteContext;

//# sourceMappingURL=useRouteContext.cjs.map