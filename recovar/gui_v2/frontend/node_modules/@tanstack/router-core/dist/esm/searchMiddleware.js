import { deepEqual } from "./utils.js";
//#region src/searchMiddleware.ts
/**
* Retain specified search params across navigations.
*
* If `keys` is `true`, retain all current params. Otherwise, copy only the
* listed keys from the current search into the next search.
*
* @param keys `true` to retain all, or a list of keys to retain.
* @returns A search middleware suitable for route `search.middlewares`.
* @link https://tanstack.com/router/latest/docs/framework/react/api/router/retainSearchParamsFunction
*/
function retainSearchParams(keys) {
	return ({ search, next }) => {
		const result = next(search);
		if (keys === true) return {
			...search,
			...result
		};
		const copy = { ...result };
		keys.forEach((key) => {
			if (!(key in copy)) copy[key] = search[key];
		});
		return copy;
	};
}
/**
* Remove optional or default-valued search params from navigations.
*
* - Pass `true` (only if there are no required search params) to strip all.
* - Pass an array to always remove those optional keys.
* - Pass an object of default values; keys equal (deeply) to the defaults are removed.
*
* @returns A search middleware suitable for route `search.middlewares`.
* @link https://tanstack.com/router/latest/docs/framework/react/api/router/stripSearchParamsFunction
*/
function stripSearchParams(input) {
	return ({ search, next }) => {
		if (input === true) return {};
		const result = { ...next(search) };
		if (Array.isArray(input)) input.forEach((key) => {
			delete result[key];
		});
		else Object.entries(input).forEach(([key, value]) => {
			if (deepEqual(result[key], value)) delete result[key];
		});
		return result;
	};
}
//#endregion
export { retainSearchParams, stripSearchParams };

//# sourceMappingURL=searchMiddleware.js.map