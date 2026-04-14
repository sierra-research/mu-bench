import js from "@eslint/js";
import globals from "globals";
import prettier from "eslint-config-prettier";

export default [
    js.configs.recommended,
    prettier,
    {
        languageOptions: {
            globals: {
                ...globals.browser,
                ...globals.node,
            },
        },
    },
    {
        ignores: ["dist/", "src/data/data.js"],
    },
];
