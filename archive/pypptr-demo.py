import asyncio
from pyppeteer import launch
async def main():
    browser = await launch()
    page = await browser.newPage()
    url = 'http://fom.yale.edu/fom/schedule?equipid=106&sd=07/26/2021'
    await page.goto(url)
    await page.screenshot({'path': 'baidu.png'})


    dimensions = await page.evaluate('''() => {
        return {
            width: document.documentElement.clientWidth,
            height: document.documentElement.clientHeight,
            deviceScaleFactor: window.devicePixelRatio,
        }
    }''')
    print(dimensions)
    # >>> {'width': 800, 'height': 600, 'deviceScaleFactor': 1}
    await browser.close()
asyncio.get_event_loop().run_until_complete(main())

