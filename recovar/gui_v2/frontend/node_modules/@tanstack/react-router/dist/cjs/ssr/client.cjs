Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
const require_RouterClient = require("./RouterClient.cjs");
exports.RouterClient = require_RouterClient.RouterClient;
var _tanstack_router_core_ssr_client = require("@tanstack/router-core/ssr/client");
Object.keys(_tanstack_router_core_ssr_client).forEach(function(k) {
	if (k !== "default" && !Object.prototype.hasOwnProperty.call(exports, k)) Object.defineProperty(exports, k, {
		enumerable: true,
		get: function() {
			return _tanstack_router_core_ssr_client[k];
		}
	});
});
