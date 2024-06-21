from anytext import aiy

if __name__ == '__main__':
    aiy_pipe = aiy.AiyAnyText()
    # test-generation
    imgs = aiy_pipe.text_generation('photo of caramel macchiato coffee on the table, top-down perspective, with "Any" "Text" written on it using cream',
                            'example_images/gen9.png',
                            seed=101)
    for i, img in enumerate(imgs):
        img.save(f'test-2-{i}.png')
