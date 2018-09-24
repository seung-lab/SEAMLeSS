import boto3
import time
import json

def make_residual_task_message(source_z, target_z, patch_bbox, mip):
  content = {
      "type": "residual_task",
      "source_z": source_z,
      "target_z": target_z,
      "patch_bbox": patch_bbox.serialize(),
      "mip": mip,
  }
  return json.dumps(content)

def make_render_task_message(z, patches, mip):
  content = {
      "type": "render_task",
      "z": z,
      "patches": [p.serialize() for p in patches],
      "mip": mip,
  }
  return json.dumps(content)

def make_downsample_task_message(z, patches, mip):
  content = {
      "type": "downsample_task",
      "z": z,
      "patches": [p.serialize() for p in patches],
      "mip": mip,
  }
  return json.dumps(content)

def make_copy_task_message(z, source, dest, patches, mip):
  content = {
      "type": "copy_task",
      "z": z,
      "source": source,
      "dest": dest,
      "patches": [p.serialize() for p in patches],
      "mip": mip,
  }
  return json.dumps(content)

class TaskHandler:
  def __init__(self, queue_name):
    # Get the service resource
    self.sqs = boto3.client('sqs')

    self.queue_name = queue_name
    self.queue_url  = self.sqs.get_queue_url(QueueName=self.queue_name)["QueueUrl"]

  def send_message(self, message_body):
    self.sqs.send_message(
      QueueUrl=self.queue_url,
      MessageBody=message_body
    )

  def get_message(self, processing_time=90):
    response = self.sqs.receive_message(
      QueueUrl=self.queue_url,
      MaxNumberOfMessages=1,
      MessageAttributeNames=[
          'All'
      ],
      VisibilityTimeout=0
    )

    if 'Messages' in response.keys() and len(response['Messages']) > 0:
      message = response['Messages'][0]
      receipt_handle = message['ReceiptHandle']
      self.sqs.change_message_visibility(
              QueueUrl=self.queue_url,
              ReceiptHandle=receipt_handle,
              VisibilityTimeout=processing_time
      )
      return message
    else:
      return None

  def delete_message(self, message):
    receipt_handle = message['ReceiptHandle']
    self.sqs.delete_message(
        QueueUrl=self.queue_url,
        ReceiptHandle=receipt_handle
    )
  def is_empty(self):
    # hashtag hackerlife
    attribute_names = ['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
    for i in range(2):
      response = self.sqs.get_queue_attributes(QueueUrl=self.queue_url,
                                               AttributeNames=attribute_names)
      for a in attribute_names:
        if int(response['Attributes'][a]) > 0:
          return False
      time.sleep(5)
    return True

  def wait_until_ready(self):
    while not self.is_empty():
      time.sleep(1)

