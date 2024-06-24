from anytext import aiy

if __name__ == '__main__':
    aiy_pipe = aiy.AiyAnyText()
    # test-generation
    imgs = aiy_pipe.text_generation('1 paper on the table, top-down perspective, with "Any2" "Text" written on it using red pen',
                            'example_images/gen9.png',
                            seed=101)
    for i, img in enumerate(imgs):
        img.save(f'test-2-{i}.png')
