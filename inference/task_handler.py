import boto3
import time
import json
import tenacity

retry = tenacity.retry(
  reraise=True, 
  stop=tenacity.stop_after_attempt(7), 
  wait=tenacity.wait_full_jitter(0.5, 60.0),
)

def make_copy_task_message(src_cv, dst_cv, src_z, dst_z, patches, mip, is_field,
                           mask_cv, mask_mip, mask_val):
  content = {
      "type": "copy_task",
      "src_cv": src_cv.serialize(),
      "dst_cv": dst_cv.serialize(),
      "src_z": src_z,
      "dst_z": dst_z,
      "patches": [p.serialize() for p in patches],
      "mip": mip,
      "is_field": is_field,
      "mask_cv": mask_cv.serialize(),
      "mask_mip": mask_mip,
      "mask_val": mask_val
  }
  return json.dumps(content)

def make_compute_field_task_message(model_path, src_cv, tgt_cv, field_cv, 
  				    src_z, tgt_z, patch_bbox, mip, pad):
  content = {                   
      "type": "compute_field",
      "model_path": model_path,
      "src_cv": src_cv.serialize(),
      "tgt_cv": tgt_cv.serialize(),
      "field_cv": field_cv.serialize(),
      "src_z": src_z,
      "tgt_z": tgt_z,
      "patch_bbox": patch_bbox.serialize(),
      "mip": mip,
      "pad": pad,
  }
  return json.dumps(content)

def make_render_task_message(src_cv, field_cv, dst_cv, src_z, field_z, dst_z, 
                             patches, src_mip, field_mip, mask_cv, mask_mip, mask_val):
  content = {
      "type": "render_task",
      "src_cv": src_cv.serialize(),
      "field_cv": field_cv.serialize(),
      "dst_cv": dst_cv.serialize(),
      "src_z": src_z,
      "field_z": field_z,
      "dst_z": dst_z,
      "patches": [p.serialize() for p in patches],
      "src_mip": mip,
      "field_mip": field_mip,
      "mask_cv": mask_cv.serialize(),
      "mask_mip": mask_mip,
      "mask_val": mask_val,
  }
  return json.dumps(content)

def make_vector_vote_task_message(pairwise_cvs, vvote_cv, z, bbox, mip, 
                                  inverse, softmin_temp, serial):
  content = {
      "type": "vector_vote_task",
      "pairwise_cvs": {k: cv.serialize() for k, cv in pairwise_cvs.items()},
      "vvote_cv": vvote_cv.serialize(),
      "z": z,
      "patch_bbox": patch_bbox.serialize(),
      "mip": mip,
      "inverse": inverse,
      "softmin_temp": softmin_temp,
      "serial": serial,
  }
  return json.dumps(content)

def make_compose_task_message(f_cv, g_cv, dst_cv, f_z, g_z, dst_z, patch_bbox, 
                              f_mip, g_mip, dst_mip):
  content = {
      "type": "compose_task",
      "f_cv": f_cv.serialize(),
      "g_cv": g_cv.serialize(),
      "f_z": f_z,
      "g_z": g_z,
      "dst_z": dst_z,
      "patch_bbox": patch_bbox.serialize(),
      "f_mip": f_mip,
      "g_mip": g_mip,
      "dst_mip": dst_mip,
  }
  return json.dumps(content)

def make_invert_field_task_message(z, src_cv, dst_cv, patch_bbox, mip, optimizer):
  content = {
      "type": "invert_task",
      "z": z,
      "src_cv": src_cv.serialize(),
      "dst_cv": dst_cv.serialize(),
      "patch_bbox": patch_bbox.serialize(),
      "mip": mip,
      "optimizer": optimizer,
  }
  return json.dumps(content)

def make_regularize_task_message(z_start, z_end, compose_start, patch_bbox, mip, sigma):
  content = {
      "type": "regularize_task",
      "z_start": z_start,
      "z_end": z_end,
      "compose_start": compose_start,
      "patch_bbox": patch_bbox.serialize(),
      "mip": mip,
      "sigma": sigma,
  }
  return json.dumps(content)


 #def make_vector_vote_task_message(z, read_F_cv, write_F_cv, patch_bbox, mip, inverse, T):
 #  content = {
 #      "type": "vector_vote_task",
 #      "z": z,
 #      #"z_end": z_range.stop,
 #      "read_F_cv": read_F_cv.serialize(),
 #      "write_F_cv": write_F_cv.serialize(),
 #      "patch_bbox": patch_bbox.serialize(),
 #      "mip": mip,
 #      "inverse": inverse,
 #      "T": T,
 #  }
 #  return json.dumps(content)


def make_prepare_task_message(z, patches, mip, start_z):
  content = {
      "type": "prepare_task",
      "z": z,
      "patches": [p.serialize() for p in patches],
      "mip": mip,
      "start_z": start_z,
  }
  return json.dumps(content)

def make_render_cv_task_message(z, field_cv, field_z, patches, mip, dst_cv, dst_z):
  content = {
      "type": "render_task_cv",
      "z": z,
      "field_cv": field_cv.serialize(),
      "field_z": field_z,
      "patches": [p.serialize() for p in patches],
      #"patches": patches.serialize(),
      "mip": mip,
      "dst_cv": dst_cv.serialize(),
      "dst_z": dst_z,
  }
  return json.dumps(content)

def make_upsample_render_rechunk_task(z_range, src_cv, field_cv, dst_cv, 
                                      patches, image_mip, field_mip):
  content = {
      "type": "upsample_render_rechunk_task",
      "z_start": z_range[0],
      "z_end": z_range[-1],
      "src_cv": src_cv.serialize(),
      "field_cv": field_cv.serialize(),
      "dst_cv": dst_cv.serialize(),
      "patches": [p.serialize() for p in patches],
      "image_mip": image_mip,
      "field_mip": field_mip,
  }
  return json.dumps(content)


def make_batch_render_message(z, field_cv, field_z, patches, mip, dst_cv,
                              dst_z, batch):
  content = {
      "type": "batch_render_task",
      "z": z,
      "field_cv": field_cv.serialize(),
      "field_z": field_z,
      "patches": [p.serialize() for p in patches],
      #"patches": patches.serialize(),
      "mip": mip,
      "dst_cv": dst_cv.serialize(),
      "dst_z": dst_z,
      "batch": batch,
  }
  return json.dumps(content)

def make_render_low_mip_task_message(z, field_cv, field_z, patches, image_mip, vector_mip, dst_cv, dst_z):
  content = {
      "type": "render_task_low_mip",
      "z": z,
      "field_cv": field_cv.serialize(),
      "field_z": field_z,
      "patches": [p.serialize() for p in patches],
      "image_mip": image_mip,
      "vector_mip": vector_mip,
      "dst_cv": dst_cv.serialize(),
      "dst_z": dst_z,
  }
  return json.dumps(content)

def make_downsample_task_message(cv, z, patches, mip):
  content = {
      "type": "downsample_task",
      "z": z,
      "cv": cv.serialize(),
      "patches": [p.serialize() for p in patches],
      #"patches": patches.serialize(),
      "mip": mip,
  }
  return json.dumps(content)

class TaskHandler:
  def __init__(self, queue_name):
    # Get the service resource
    self.sqs = boto3.client('sqs')

    self.queue_name = queue_name
    self.queue_url  = self.sqs.get_queue_url(QueueName=self.queue_name)["QueueUrl"]

  @retry
  def send_message(self, message_body):
    attribute_names = ['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
    threshold = 100000
    # while(True):
    #     response = self.sqs.get_queue_attributes(QueueUrl=self.queue_url,
    #                                              AttributeNames=attribute_names)
    #     Message_num = int(response['Attributes']['ApproximateNumberOfMessages'])
    #     if Message_num > threshold:
    #         print("Message number is", Message_num, "sleep")
    #         time.sleep(3)
    #     else: 
    #         self.sqs.send_message(QueueUrl=self.queue_url, 
    #                               MessageBody=message_body)
    #         break
    self.sqs.send_message(QueueUrl=self.queue_url, MessageBody=message_body)

  @retry
  def get_message(self, processing_time=90):
    response = self.sqs.receive_message(
      QueueUrl=self.queue_url,
      MaxNumberOfMessages=1,
      MessageAttributeNames=[
          'All'
      ],
      VisibilityTimeout=processing_time
    )

    if 'Messages' in response.keys() and len(response['Messages']) > 0:
      message = response['Messages'][0]
      receipt_handle = message['ReceiptHandle']
      # self.sqs.change_message_visibility(
      #         QueueUrl=self.queue_url,
      #         ReceiptHandle=receipt_handle,
      #         VisibilityTimeout=processing_time
      # )
      return message
    else:
      return None
  
  @retry
  def delete_message(self, message):
    receipt_handle = message['ReceiptHandle']
    self.sqs.delete_message(
        QueueUrl=self.queue_url,
        ReceiptHandle=receipt_handle
    )

  @retry
  def is_empty(self):
    # hashtag hackerlife
    attribute_names = ['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
    for i in range(2):
      response = self.sqs.get_queue_attributes(QueueUrl=self.queue_url,
                                               AttributeNames=attribute_names)
      print(response)
      for a in attribute_names:
        if int(response['Attributes'][a]) > 0:
          return False
      time.sleep(5)
    print("donot wait since it")
    return True
  
  def purge_queue(self):
    self.sqs.purge_queue(QueueUrl=self.queue_url)

  def wait_until_ready(self):
    time.sleep(20)
    while not self.is_empty():
      time.sleep(5)

