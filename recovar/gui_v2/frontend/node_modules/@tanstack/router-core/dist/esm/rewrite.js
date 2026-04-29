import { joinPaths, trimPath } from "./path.js";
//#region src/rewrite.ts
/** Compose multiple rewrite pairs into a single in/out rewrite. */
/** Compose multiple rewrite pairs into a single in/out rewrite. */
function composeRewrites(rewrites) {
	return {
		input: ({ url }) => {
			for (const rewrite of rewrites) url = executeRewriteInput(rewrite, url);
			return url;
		},
		output: ({ url }) => {
			for (let i = rewrites.length - 1; i >= 0; i--) url = executeRewriteOutput(rewrites[i], url);
			return url;
		}
	};
}
/** Create a rewrite pair that strips/adds a basepath on input/output. */
/** Create a rewrite pair that strips/adds a basepath on input/output. */
function rewriteBasepath(opts) {
	const trimmedBasepath = trimPath(opts.basepath);
	const normalizedBasepath = `/${trimmedBasepath}`;
	const normalizedBasepathWithSlash = `${normalizedBasepath}/`;
	const checkBasepath = opts.caseSensitive ? normalizedBasepath : normalizedBasepath.toLowerCase();
	const checkBasepathWithSlash = opts.caseSensitive ? normalizedBasepathWithSlash : normalizedBasepathWithSlash.toLowerCase();
	return {
		input: ({ url }) => {
			const pathname = opts.caseSensitive ? url.pathname : url.pathname.toLowerCase();
			if (pathname === checkBasepath) url.pathname = "/";
			else if (pathname.startsWith(checkBasepathWithSlash)) url.pathname = url.pathname.slice(normalizedBasepath.length);
			return url;
		},
		output: ({ url }) => {
			url.pathname = joinPaths([
				"/",
				trimmedBasepath,
				url.pathname
			]);
			return url;
		}
	};
}
/** Execute a location input rewrite if provided. */
/** Execute a location input rewrite if provided. */
function executeRewriteInput(rewrite, url) {
	const res = rewrite?.input?.({ url });
	if (res) {
		if (typeof res === "string") return new URL(res);
		else if (res instanceof URL) return res;
	}
	return url;
}
/** Execute a location output rewrite if provided. */
/** Execute a location output rewrite if provided. */
function executeRewriteOutput(rewrite, url) {
	const res = rewrite?.output?.({ url });
	if (res) {
		if (typeof res === "string") return new URL(res);
		else if (res instanceof URL) return res;
	}
	return url;
}
//#endregion
export { composeRewrites, executeRewriteInput, executeRewriteOutput, rewriteBasepath };

//# sourceMappingURL=rewrite.js.map