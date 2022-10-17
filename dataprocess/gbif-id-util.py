import requests, argparse, httpx
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List
import json, os, sys, asyncio
import functional as pyfun

GBIF_API_URL = 'https://api.gbif.org/v1/species/'

@dataclass
#class InsectInfo(frozen=True):
class InsectInfo():
    species: str = field()
    canonical_name: str = field()
    vernacular_name: str = field()
    full_description: Dict = field(hash=False, repr=False)


# httpx response: https://www.python-httpx.org/api/#response
# requests response: https://requests.readthedocs.io/en/latest/api/#requests.Response
def handle_http_response(res) -> Exception | Dict:
    match res.status_code:
        case 200:
            tmp = res.json()
            # print(type(tmp))
            # print(f'species: {species}, canonical name: {canonical_name}, vernacular name: {vernacular_name}')
            return tmp
        case code if code >=400 and code <= 451:
            return Exception(f'url {res.request.url} not found')
        case _:
            return Exception(f'response code: {res.status_code}')


def retrieve_insect_info(gbif_id: str) -> Exception | Dict:
    req_url = f'{GBIF_API_URL}{gbif_id}'
    # print(f'request url: {req_url}')
    gbif_res = requests.get(req_url)
    return handle_http_response(gbif_res)


def extract_insect_info(content: Dict) -> InsectInfo:
    species = content['species']
    canonical_name = content['canonicalName']
    vernacular_name = content.get('vernacularName', 'N/A')
    insect_info = InsectInfo(species, canonical_name, vernacular_name, content)

    return insect_info


def write_json_description(path: str, gbif_id: str, content: Dict):
    fn = os.path.join(path, f'{gbif_id}/description.json')
    print(f'file name is {fn}')
    with open(fn, 'w') as fh:
        fh.write(json.dumps(content, indent=2))


def retreve_all_insect_info_requests(id_list: List[str], folder: str):
        #.map(lambda d: (d['key'], d['canonicalName'])) 
    pyfun.pseq(id_list, partition_size=8) \
        .map(lambda id: retrieve_insect_info(id)) \
        .filter(lambda r: isinstance(r, Dict)) \
        .map(lambda d: (d['key'], d)) \
        .for_each(lambda t: write_json_description(folder, t[0], t[1]))


def retreve_all_insect_info_httpx(id_list: List[str], folder: str):
    async def load(ids: List[str]):
        async with httpx.AsyncClient(timeout=10.0) as client:
            #tasks = [get_insect(id, client) for id in id_list]
            tasks = [client.get(f'{GBIF_API_URL}{id}') for id in id_list]
            reqs = await asyncio.gather(*tasks)

        return reqs

    reqs_list = asyncio.run(load(id_list))
    print(reqs_list)
    pyfun.seq(reqs_list) \
        .map(handle_http_response) \
        .filter(lambda r: isinstance(r, Dict)) \
        .map(lambda d: (d['key'], d)) \
        .for_each(lambda t: write_json_description(folder, t[0], t[1]))


def load_folder_names(folder: str) -> List[str]:
    subdirs = [f.name for f in os.scandir(folder) if f.is_dir()]
    return subdirs


def inset_gbif_test(id: str):
    info = retrieve_insect_info(id)
    print(info)
    match info:
        case Exception() as info:
            print(info)
        case dict() as info:
            insect = extract_insect_info(info)
            print(insect)
            print(insect.full_description)
            #write_json_description(path, gbif_id, insect.full_description)


def main(args):
    match args.command:
        case 'search':
            gbif_id = args.id
            # gbif_id = '123a' # non-exist id
            inset_gbif_test(gbif_id)
        case 'pull_all':
            labels = load_folder_names(args.dir)
            # retreve_all_insect_info_requests(labels, folder=args.dir)
            retreve_all_insect_info_httpx(labels, folder=args.dir)
        case 'test':
            dirs = load_folder_names(args.dir)
            print(dirs)
        case _:
            print(f'unknown command [{args.command}]')

if __name__ == '__main__':
    print(f'python version is {sys.version_info}')
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 10):
        sys.exit('this program needs python 3.10 and above to run')

    # https://towardsdatascience.com/a-simple-guide-to-command-line-arguments-with-argparse-6824c30ab1c3
    aparser = argparse.ArgumentParser(description='gbif info utility')
    sub = aparser.add_subparsers(dest='command', help='command to run', required=True)
    search_arg = sub.add_parser('search', help='search a single species by gbif id')
    search_arg.add_argument('-i', '--id', type=str, help='insect id to search',
                          required=True)
    pull_arg = sub.add_parser('pull_all', help='pull all species and save into folders')
    pull_arg.add_argument('-d', '--dir', type=str, help='file directory',
                          required=True)
    test_arg = sub.add_parser('test', help='test functions')
    test_arg.add_argument('-d', '--dir', type=str, help='file directory',
                          required=True)
    # aparser.add_argument('-s', '--start', type=int, help='start page',
    #                      required=True)
    # aparser.add_argument('-o', '--output', type=str, help='output file name',
    #                      required=True)
    targs = aparser.parse_args()

    main(targs)
