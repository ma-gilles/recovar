import { useLayoutEffect } from "./utils.js";
import { useRouter } from "./useRouter.js";
import * as React$1 from "react";
//#region src/useNavigate.tsx
/**
* Imperative navigation hook.
*
* Returns a stable `navigate(options)` function to change the current location
* programmatically. Prefer the `Link` component for user-initiated navigation,
* and use this hook from effects, callbacks, or handlers where imperative
* navigation is required.
*
* Options:
* - `from`: Optional route base used to resolve relative `to` paths.
*
* @returns A function that accepts `NavigateOptions`.
* @link https://tanstack.com/router/latest/docs/framework/react/api/router/useNavigateHook
*/
function useNavigate(_defaultOpts) {
	const router = useRouter();
	return React$1.useCallback((options) => {
		return router.navigate({
			...options,
			from: options.from ?? _defaultOpts?.from
		});
	}, [_defaultOpts?.from, router]);
}
/**
* Component that triggers a navigation when rendered. Navigation executes
* in an effect after mount/update.
*
* Props are the same as `NavigateOptions` used by `navigate()`.
*
* @returns null
* @link https://tanstack.com/router/latest/docs/framework/react/api/router/navigateComponent
*/
function Navigate(props) {
	const router = useRouter();
	const navigate = useNavigate();
	const previousPropsRef = React$1.useRef(null);
	useLayoutEffect(() => {
		if (previousPropsRef.current !== props) {
			navigate(props);
			previousPropsRef.current = props;
		}
	}, [
		router,
		props,
		navigate
	]);
	return null;
}
//#endregion
export { Navigate, useNavigate };

//# sourceMappingURL=useNavigate.js.map