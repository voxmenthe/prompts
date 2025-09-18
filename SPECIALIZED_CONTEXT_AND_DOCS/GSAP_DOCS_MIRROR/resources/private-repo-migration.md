# Moving off the private repo

info

Thanks to Webflow's support, The core GSAP library and all the plugins are now 100% free for everyone! This page covers how to move away from the private repository.

If you're one of our wonderful Club GSAP members - you may be wondering what you need to change in your project to continue using the GSAP files you know and love.

The good news is that it's nice and simple from now on! No more tokenised access to a private repository, no more .npmrc files or fiddly yarn install issues. 🥳

### NPM / Yarn (or other package managers)[​](#npm--yarn-or-other-package-managers "Direct link to NPM / Yarn (or other package managers)")

All the GSAP files are now publicly available on npm for everyone to use - just install GSAP with your package manager of choice.

* npm
* Yarn Classic
* Yarn Berry

1. Remove the following from your .npmrc -

   .npmrc

   ```
   always-auth=true
   //npm.greensock.com/:_authToken=24e2f3ca-63e0-4b93-b68e-a4695672c571
   @gsap:registry=https://npm.greensock.com
   ```

2. Remove the Club GSAP dependency.

Take a look in your package.json to check which tier of Club GSAP you have installed.

package.json

```
{
	"name": "gsap-bonus",
	"version": "0.1.0",
	"private": true,
	"dependencies": {
		"@gsap/business": "^3.13.0",
		"next": "14.2.3",
		"react": "^18",
		"react-dom": "^18"
	}
}
```

In this case we have the **business** package installed so we'll run the following in terminal.

bash

```
npm uninstall @gsap/business
```

3. Install the public gsap package

Now just install GSAP from the public repository, and you're done!

bash

```
npm install gsap
```

1. Remove the following from your .npmrc -

.npmrc

```
always-auth=true
//npm.greensock.com/:_authToken=24e2f3ca-63e0-4b93-b68e-a4695672c571
@gsap:registry=https://npm.greensock.com/
```

2. Remove the Club GSAP dependency.

Take a look in your package.json to check which tier of Club GSAP you have installed.

package.json

```
{
	"name": "gsap-bonus",
	"version": "0.1.0",
	"private": true,
	"dependencies": {
		"@gsap/business": "^3.13.0",
		"next": "14.2.3",
		"react": "^18",
		"react-dom": "^18"
	}
}
```

In this case we have the **business** package installed so we'll run the following in terminal.

bash

```
yarn remove @gsap/business
```

3. Install the public gsap package

Now just install GSAP from the public repository, and you're done!

bash

```
yarn add gsap
```

1. Remove the following from your .yarnrc.yml -

.yarnrc.yml

```
unsafeHttpWhitelist:
  - "npm.greensock.com"

npmScopes:
  gsap:
    npmRegistryServer: "https://npm.greensock.com"
    npmAuthToken: 24e2f3ca-63e0-4b93-b68e-a4695672c571
```

2. Remove the Club GSAP dependency.

Take a look in your package.json to check which tier of Club GSAP you have installed.

package.json

```
{
	"name": "gsap-bonus",
	"version": "0.1.0",
	"private": true,
	"dependencies": {
		"@gsap/business": "^3.13.0",
		"next": "14.2.3",
		"react": "^18",
		"react-dom": "^18"
	}
}
```

In this case we have the **business** package installed so we'll run the following in terminal.

bash

```
yarn remove @gsap/business
```

3. Install the public gsap package

Now just install GSAP from the public repository, and you're done!

bash

```
yarn add gsap
```

<br />

Cached files?

If you run into issues, you may need to delete `node_modules` or lockfiles to ensure that nothing is referencing the old Club files.

bash

```
# npm
rm -rf node_modules package-lock.json && npm install

# yarn classic
rm -rf node_modules yarn.lock && yarn install

# yarn berry (with PnP)
yarn install --immutable
```

For Yarn Berry: you may want to also check .yarn/cache and .yarn/unplugged folders if you had the Club package linked or cached.

### Script Tags[​](#script-tags "Direct link to Script Tags")

If you're using script tags in your project, there's nothing you *need* to change. Your self-hosted Club Plugins will continue to work! But if you would like the performance benefits of using a CDN, visit the [install helper](https://gsap.com/docs/v3/Installation?tab=cdn\&module=esm\&require=false) to grab your new links.

#### Still stuck? Pop by the [GSAP forums](https://gsap.com/community/) and we'll give you a hand. 💚[​](#still-stuck-pop-by-the-gsap-forums-and-well-give-you-a-hand- "Direct link to still-stuck-pop-by-the-gsap-forums-and-well-give-you-a-hand-")
