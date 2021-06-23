# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# def get_resolution(original_resolution):
#   """Takes (H,W) and returns (precrop, crop)."""
#   area = original_resolution[0] * original_resolution[1]
#   return (160, 128) if area < 96*96 else (512, 480)


known_dataset_sizes = {
  'cifar10': (32, 32),
  'cifar100': (32, 32),
  'oxford_iiit_pet': (224, 224),
  'oxford_flowers102': (224, 224),
  'imagenet2012': (224, 224),
  # TODO: Specify image size of custom dataset here
  'logo_2k': (224, 224),
  'targetlist': (224, 224),
  'web':(10, 10),
#   'screenshot': (256, 256)
#   'screenshot': (1080, 1920), # HxW
    'screenshot': (256, 512)
}


def get_resolution_from_dataset(dataset):
    if dataset not in known_dataset_sizes:
        raise ValueError(f"Unsupported dataset {dataset}. Add your own here :)")
    return known_dataset_sizes[dataset]


def get_mixup(dataset_size):
    return 0.0 if dataset_size < 20_000 else 0.1


# Not used
def get_schedule(dataset_size):
    if dataset_size < 20_000:
        return [400, 800, 1200, 1600, 2000]
    elif dataset_size < 500_000:
        return [5000, 30000, 60000, 90000, 10_0000]
    else:
        return [5000, 30000, 60000, 90000, 10_0000]


def get_lr(step, dataset_size, base_lr):
    """Returns learning-rate for `step` or None at the end."""
    supports = get_schedule(dataset_size)
    # Do not use learning rate warmup
    if step < supports[0]:
        return base_lr
    # End of training
    elif step >= supports[-1]:
        return None
    # Staircase decays by factor of 10
    else:
        for s in supports[1:]:
            if s < step:
                base_lr /= 10
    return base_lr

################## Schedule designed for finetuning ###################################################### 

def get_schedule_finetune(dataset_size, batch_size):
    epoch_steps = dataset_size // batch_size # one epoch takes xxx iterations
    return [epoch_steps*5] # only finetune for 5 epochs

def get_lr_finetune(step, dataset_size, base_lr, batch_size):
    """Returns learning-rate for `step` or None at the end."""
    supports = get_schedule_finetune(dataset_size, batch_size)
    # Do not use learning rate warmup
    if step < supports[0]:
        return base_lr
    # End of training
    else:
        return None

