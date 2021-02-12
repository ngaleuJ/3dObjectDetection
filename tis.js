(function () {
jQuery(function () {

	return jQuery('[data-toggle]').on('click', function () {
		var toggle;
		toggle = jQuery(this).addClass('active').attr('data-toggle');
		jQuery(this).siblings('[data-toggle]').removeClass('active');
		return jQuery('.surveys').removeClass('grid list').addClass(toggle);
	});

});
}.call(this));


jQuery(document).ready(function(){
	if(jQuery('.it_epoll_frontend_close_btn').length){
		jQuery('.it_epoll_frontend_close_btn').on('click',function(){
		jQuery('.it_epoll_modal_inner').toggleClass('it_epoll_modal_inner_hide');
		jQuery('.it_epoll_modal_inner').fadeOut(400);
		jQuery('.it_epoll_modal_outer').fadeOut(500);
		});	
	}
	

	jQuery('.it_epoll_surveys .it_epoll_survey-item').each(function(){
		jQuery(this).find('#it_epoll_otp_confirmation_form').validate({

			submitHandler: function(form){
				var it_epoll_item = jQuery(form).parent().parent();

			jQuery(it_epoll_item).parent().find('.it_epoll_survey-item').each(function(){
				var it_epoll_multivote = jQuery(this).find('#it_epoll_multivoting').val();
				if(!it_epoll_multivote){
					jQuery(this).find('#it_epoll_survey-confirm-button').val('...');

					jQuery(this).find('#it_epoll_survey-confirm-button').attr('disabled','yes');
				}
				
			});



			jQuery(it_epoll_item).find('.it_epoll_spinner').fadeIn();

				jQuery.ajax({
					          url: it_epoll_ajax_obj.ajax_url,
					          data: {'action':'it_epoll_vote_confirm', 'data':jQuery(form).serialize(),'option_id': jQuery(it_epoll_item).find('#it_epoll_survey-item-id').val(),
								'poll_id': jQuery(it_epoll_item).find('#it_epoll_poll-id').val()},
					        
					          beforeSend: function (request)
									{
										jQuery('.it_epoll_card_back').removeClass('it_epoll_card_back_visible');
                                    },
							type: 'POST',
							success: function(data){

								var it_epoll_json = jQuery.parseJSON(data);
								
								if(it_epoll_json.voting_status =='done'){
									var it_epoll_multivote = 0;
									jQuery(it_epoll_item).parent().find('.it_epoll_survey-item').each(function(){
											it_epoll_multivote = jQuery(this).find('#it_epoll_multivoting').val();
											
											if(it_epoll_multivote == 0){
												jQuery(this).find('#it_epoll_survey-confirm-button').val('...');
												//console.log(jQuery(this).find('#it_epoll_survey-confirm-button'));
												jQuery(this).find('#it_epoll_survey-confirm-button').attr('disabled','yes');
											}
											
										});

									jQuery(it_epoll_item).find('.it_epoll_survey-progress-fg').fadeIn();
									jQuery(it_epoll_item).find('.it_epoll_survey-progress-fg').attr('style','width:'+it_epoll_json.total_vote_percentage+'%');
									jQuery(it_epoll_item).find('.it_epoll_survey-progress-label').text(it_epoll_json.total_vote_percentage+'%');
									jQuery(it_epoll_item).find('.it_epoll_survey-completes').text(it_epoll_json.total_opt_vote_count+' / '+it_epoll_json.total_vote_count);		
									jQuery(it_epoll_item).parent().find('.it_epoll_survey-item').each(function(){
										it_epoll_multivote = jQuery(this).find('#it_epoll_multivoting').val();
									
											if(it_epoll_multivote == 0){
						        				jQuery(this).find('#it_epoll_survey-vote-button').addClass('it_epoll_scale_hide');
						        				}
						    				});
									setTimeout(function(){
										jQuery(it_epoll_item).find('#it_epoll_survey-confirm-button').addClass('it_epoll_green_gradient');
										jQuery(it_epoll_item).find('#it_epoll_survey-confirm-button').val("Voted");
										jQuery(it_epoll_item).find('#it_epoll_survey-confirm-button').attr("disabled","yes");
										jQuery(it_epoll_item).find('.it_epoll_spinner').toggleClass("it_epoll_spinner_stop");	
										jQuery(it_epoll_item).find('.it_epoll_spinner').toggleClass("it_epoll_drawn");
									},300);
								}else{
									jQuery(it_epoll_item).find('#it_epoll_otp_confirm_form').addClass('it_epoll_card_back_visible');
									jQuery(it_epoll_item).find('.it_epoll_short_code').text("Invalid OTP Code");
								}
							}
							});	
		}

		});
		jQuery(this).find('#it_epoll_survey_confirmation_form').validate({

			submitHandler: function(form){
				var it_epoll_item = jQuery(form).parent().parent(); 
				jQuery(it_epoll_item).parent().find('.it_epoll_survey-item').each(function(){
				//jQuery(this).find('#it_epoll_survey-confirm-button').val('...');
				//jQuery(this).find('#it_epoll_survey-confirm-button').attr('disabled','yes');
			});



			jQuery(it_epoll_item).find('.it_epoll_spinner').fadeIn();

				jQuery.ajax({
					          url: it_epoll_ajax_obj.ajax_url,
					          data: {'action':'it_epoll_vote_by_form', 'data':jQuery(form).serialize(),'option_id': jQuery(it_epoll_item).find('#it_epoll_survey-item-id').val(),
								'poll_id': jQuery(it_epoll_item).find('#it_epoll_poll-id').val()},
					        
					          beforeSend: function (request)
									{
										jQuery('.it_epoll_card_back').removeClass('it_epoll_card_back_visible');
                                    },
							type: 'POST',
							success: function(data){

								var it_epoll_json = jQuery.parseJSON(data);
								//console.log("done"+it_epoll_json);
								if(it_epoll_json.voting_status =='done'){
										var it_epoll_multivote = 0;
										jQuery(it_epoll_item).parent().find('.it_epoll_survey-item').each(function(){
											it_epoll_multivote = jQuery(this).find('#it_epoll_multivoting').val();
											//console.log(it_epoll_multivote);
											if(it_epoll_multivote == 0){
												jQuery(this).find('#it_epoll_survey-confirm-button').val('...');

												jQuery(this).find('#it_epoll_survey-confirm-button').attr('disabled','yes');
											}
											
										});

									jQuery(it_epoll_item).find('.it_epoll_survey-progress-fg').fadeIn();
									jQuery(it_epoll_item).find('.it_epoll_survey-progress-fg').attr('style','width:'+it_epoll_json.total_vote_percentage+'%');
									jQuery(it_epoll_item).find('.it_epoll_survey-progress-label').text(it_epoll_json.total_vote_percentage+'%');
									jQuery(it_epoll_item).find('.it_epoll_survey-completes').text(it_epoll_json.total_opt_vote_count+' / '+it_epoll_json.total_vote_count);		
									
									setTimeout(function(){
										jQuery(it_epoll_item).find('#it_epoll_survey-confirm-button').addClass('it_epoll_green_gradient');
										jQuery(it_epoll_item).find('#it_epoll_survey-confirm-button').val("Voted");
										jQuery(it_epoll_item).find('#it_epoll_survey-confirm-button').attr("disabled","yes");
										jQuery(it_epoll_item).find('.it_epoll_spinner').toggleClass("it_epoll_spinner_stop");	
										jQuery(it_epoll_item).find('.it_epoll_spinner').toggleClass("it_epoll_drawn");
									},300);
								}else if(it_epoll_json.voting_status == 'otp_step'){
									if(it_epoll_json.otp_email){
									jQuery(it_epoll_item).find('#it_epoll_otp_confirmation_form #email_otp_confirm_input').val(it_epoll_json.otp_email);
										
									}
									if(it_epoll_json.otp_phone){
									jQuery(it_epoll_item).find('#it_epoll_otp_confirmation_form #phone_otp_confirm_input').val(it_epoll_json.otp_phone);
									}
									jQuery(it_epoll_item).find('#it_epoll_otp_confirm_form').addClass('it_epoll_card_back_visible');
								}else{
									jQuery(it_epoll_item).find('#it_epoll_otp_confirm_form').html('<span style="background: #fc7977; align-items: center;display: flex; color:#fff;border-radius:6px; padding:10px;width:100%;margin:0;">You Are Already Participated!</span>');
									jQuery(it_epoll_item).find('#it_epoll_otp_confirm_form').addClass('it_epoll_card_back_visible');
								}
							}
							});	
		}

		});
	});
	});