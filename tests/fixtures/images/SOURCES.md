# Test images

Each test image is a work of a US federal agency, so it is in the public domain in the US. We checked each license page on 2026-11-06. `scripts/make_test_images.py` makes the files: it downloads the source file, checks its SHA-1 against the Wikimedia Commons API, crops the center, resizes with Pillow LANCZOS to 1024 px on the long side and saves a JPEG with quality 92. The script removes all metadata.

The NASA media guidelines ask users not to suggest an endorsement by NASA. This repo uses the photos only as test inputs. NASA and the USDA Forest Service do not endorse it.

| File | Stored size | SHA-256 of the stored file | Source page | Author | License | Source size | SHA-1 of the source | Crop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `iss-kibo-lab.jpg` | 1024 × 576 | `949983ccb96f975b3db569876a1f5990e15b2fe4eac4c345d1d548ba379c102e` | https://commons.wikimedia.org/wiki/File:ISS-22_crew_members_in_the_Kibo_lab.jpg | NASA | Public domain (PD NASA) | 4288 × 2848 | `564aa44ebf0db3683e2b5b86d48622475336f496` | center 4288 × 2412 |
| `forest.jpg` | 1024 × 768 | `5de0c07ff69642d8d082d3ded7f8da15fd1f206481246898c73ed8d6f41e6e2d` | https://commons.wikimedia.org/wiki/File:South_Fork_Amercian_River_Cohesive_Strategy_Field_Trip_(36013609855).jpg | USDA Forest Service, Pacific Southwest Region 5 | Public domain (PD USDA) | 4608 × 3456 | `3116c98b7974dc25134658e1140394410962efcb` | none |
| `iss-conference.jpg` | 768 × 1024 | `8069ac0a4e0d503aa1a7e788b2e32d2d377551254a8a5cb30742f54fb3e07ac4` | https://commons.wikimedia.org/wiki/File:Expedition_75_crew_members_gather_for_a_daily_planning_conference_(iss075e0162800).jpg | NASA Johnson Space Center | Public domain (PD NASA) | 3024 × 4032 | `9396757b90e0e888ec02bdcd5614ac6c69496e7f` | none |

Sizes are width × height in pixels.
