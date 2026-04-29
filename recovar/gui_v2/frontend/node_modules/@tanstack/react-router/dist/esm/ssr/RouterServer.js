import { RouterProvider } from "../RouterProvider.js";
import "react";
import { jsx } from "react/jsx-runtime";
//#region src/ssr/RouterServer.tsx
function RouterServer(props) {
	return /* @__PURE__ */ jsx(RouterProvider, { router: props.router });
}
//#endregion
export { RouterServer };

//# sourceMappingURL=RouterServer.js.map