const defineViteConfig = require('vite').defineConfig
const react = require('@vitejs/plugin-react').default
const nodeResolve = require('@rollup/plugin-node-resolve').default
const path = require('path')
const viteExternalsPlugin = require('vite-plugin-externals').viteExternalsPlugin
const { FIFTYONE_DIR } = process.env
const dir = __dirname
const IS_DEV = process.env.IS_DEV === 'true'

function fiftyonePlugin() {
  if (!FIFTYONE_DIR) {
    throw new Error(
      `FIFTYONE_DIR environment variable not set. This is required to resolve @fiftyone/* imports.`
    )
  }

  return {
    name: 'fiftyone-rollup',
    resolveId: {
      order: 'pre',
      async handler(source) {
        if (source.startsWith('@fiftyone')) {
          const pkg = source.split('/')[1]
          const modulePath = `${FIFTYONE_DIR}/app/packages/${pkg}`
          return this.resolve(modulePath, source, { skipSelf: true })
        }
        return null
      }
    }
  }
}

const package = require(`${dir}/package.json`)
module.exports = defineViteConfig({
  mode: 'development',
  plugins: [
    fiftyonePlugin(),
    nodeResolve(),
    react(),
    viteExternalsPlugin({
      react: 'React',
      'react-dom': 'ReactDOM',
      recoil: 'recoil',
      '@fiftyone/state': '__fos__',
      '@fiftyone/operators': '__foo__',
      '@fiftyone/components': '__foc__',
      '@fiftyone/utilities': '__fou__',
      '@mui/material': '__mui__' // use mui from fiftyone
    })
  ],
  build: {
    minify: IS_DEV ? false : true,
    lib: {
      entry: path.join(dir, package.main),
      name: package.name,
      fileName: (format) => `index.${format}.js`,
      formats: ['umd']
    }
  },
  define: {
    'process.env.NODE_ENV': '"development"'
  },
  optimizeDeps: {
    exclude: ['react', 'react-dom']
  }
})
