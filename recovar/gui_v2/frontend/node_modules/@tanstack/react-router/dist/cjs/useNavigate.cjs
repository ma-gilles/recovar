const require_runtime = require("./_virtual/_rolldown/runtime.cjs");
const require_utils = require("./utils.cjs");
const require_useRouter = require("./useRouter.cjs");
let react = require("react");
react = require_runtime.__toESM(react);
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
	const router = require_useRouter.useRouter();
	return react.useCallback((options) => {
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
	const router = require_useRouter.useRouter();
	const navigate = useNavigate();
	const previousPropsRef = react.useRef(null);
	require_utils.useLayoutEffect(() => {
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
exports.Navigate = Navigate;
exports.useNavigate = useNavigate;

//# sourceMappingURL=useNavigate.cjs.map